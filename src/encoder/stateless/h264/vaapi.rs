// Copyright 2024 The ChromiumOS Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

//! VA-API stateless H.264 encoder backend.
//!
//! ## Packed headers
//!
//! A driver that advertises `VA_ENC_PACKED_HEADER_SLICE` (and the sequence and picture
//! headers) through `VAConfigAttribEncPackedHeaders` expects the application to supply them
//! (`va.h`: "Application must provide it to driver once this flag is returned"), and writes
//! what it is given. Mesa's radeonsi, for one, builds every slice NAL unit from the packed slice
//! header it parsed: `nal_ref_idc`, `nal_unit_type`, `frame_num`, `idr_pic_id` and the POC lsb
//! all come from it, so without one each slice starts with the header byte `0x00`. It reads
//! `entropy_coding_mode_flag` from the packed PPS to parse that slice header, too.
//!
//! So this backend takes the HEVC backend's all-three-or-none rule
//! ([`decide_packed_headers`](crate::backend::vaapi::encoder::decide_packed_headers)), as
//! Chromium's VA-API H.264 encoder does: where the driver advertises SEQUENCE, PICTURE and
//! SLICE, the IDR packs the SPS (beside the sequence parameter buffer) and the PPS, the first
//! frame after the PPS changed packs the PPS
//! ([`ParameterSets`](crate::encoder::stateless::h264::ParameterSets)), and every frame packs
//! its slice header, NAL unit header included. The coded buffer then holds the whole access
//! unit and the predictor writes no parameter set of its own (`emit_parameter_sets` is
//! cleared). Elsewhere the driver writes the slice headers and the predictor the parameter
//! sets, in band.
//!
//! The buffers go in FFmpeg's order (`vaapi_encode.c`): the sequence and picture parameters,
//! the rate control, the packed SPS and PPS, then the slice's packed header and its slice
//! parameters. A driver that parses the packed slice header (radeonsi) needs the picture
//! parameters (whether the picture is an IDR) and the PPS before it.

use std::any::Any;
use std::borrow::Borrow;
use std::rc::Rc;

use anyhow::Context;
use libva::BufferType;
use libva::Display;
use libva::EncCodedBuffer;
use libva::EncPackedHeaderType;
use libva::EncPictureParameter;
use libva::EncPictureParameterBufferH264;
use libva::EncSequenceParameter;
use libva::EncSequenceParameterBufferH264;
use libva::EncSliceParameter;
use libva::EncSliceParameterBufferH264;
use libva::H264EncFrameCropOffsets;
use libva::H264EncPicFields;
use libva::H264EncSeqFields;
use libva::H264VuiFields;
use libva::Picture;
use libva::PictureH264;
use libva::Surface;
use libva::SurfaceMemoryDescriptor;
use libva::VAEntrypoint::VAEntrypointEncSlice;
use libva::VAEntrypoint::VAEntrypointEncSliceLP;
use libva::VAProfile;
use libva::VA_INVALID_ID;
use libva::VA_PICTURE_H264_LONG_TERM_REFERENCE;
use libva::VA_PICTURE_H264_SHORT_TERM_REFERENCE;

use crate::backend::vaapi::encoder::query_packed_headers;
use crate::backend::vaapi::encoder::rate_control_to_va_rc;
use crate::backend::vaapi::encoder::tunings_to_libva_rc;
use crate::backend::vaapi::encoder::CodedOutputPromise;
use crate::backend::vaapi::encoder::Reconstructed;
use crate::backend::vaapi::encoder::VaapiBackend;
use crate::codec::h264::parser::Pps;
use crate::codec::h264::parser::Profile;
use crate::codec::h264::parser::SliceHeader;
use crate::codec::h264::parser::Sps;
use crate::codec::h264::synthesizer::Synthesizer;
use crate::encoder::h264::EncoderConfig;
use crate::encoder::h264::H264;
use crate::encoder::stateless::h264::predictor::MAX_QP;
use crate::encoder::stateless::h264::predictor::MIN_QP;
use crate::encoder::stateless::h264::BackendRequest;
use crate::encoder::stateless::h264::DpbEntry;
use crate::encoder::stateless::h264::DpbEntryMeta;
use crate::encoder::stateless::h264::IsReference;
use crate::encoder::stateless::h264::StatelessEncoder;
use crate::encoder::stateless::h264::StatelessH264EncoderBackend;
use crate::encoder::stateless::ReadyPromise;
use crate::encoder::stateless::StatelessBackendError;
use crate::encoder::stateless::StatelessBackendResult;
use crate::encoder::stateless::StatelessVideoEncoderBackend;
use crate::encoder::EncodeResult;
// Only the `#[cfg(test)]` module references `RateControl` directly now that
// the RC→VA_RC map moved to `rate_control_to_va_rc`; gate the import so a
// non-test build doesn't see it as unused.
#[cfg(test)]
use crate::encoder::RateControl;
use crate::video_frame::VideoFrame;
use crate::BlockingMode;
use crate::Fourcc;
use crate::Resolution;

type Request<'l, H> = BackendRequest<H, Reconstructed>;

impl<M, H> StatelessVideoEncoderBackend<H264> for VaapiBackend<M, H>
where
    M: SurfaceMemoryDescriptor,
    H: std::borrow::Borrow<Surface<M>> + 'static,
{
    type Picture = H;
    type Reconstructed = Reconstructed;
    type CodedPromise = CodedOutputPromise<M, H>;
    type ReconPromise = ReadyPromise<Self::Reconstructed>;
}

impl<M, H> VaapiBackend<M, H>
where
    M: SurfaceMemoryDescriptor,
    H: std::borrow::Borrow<Surface<M>> + 'static,
{
    /// Builds an invalid [`libva::PictureH264`]. This is usually a place
    /// holder to fill staticly sized array.
    fn build_invalid_va_h264_pic_enc() -> libva::PictureH264 {
        libva::PictureH264::new(libva::VA_INVALID_ID, 0, libva::VA_PICTURE_H264_INVALID, 0, 0)
    }

    /// Builds [`libva::PictureH264`] from `frame`
    fn build_h264_pic(surface: &Reconstructed, meta: &DpbEntryMeta) -> PictureH264 {
        let flags = match meta.is_reference {
            IsReference::No => 0,
            IsReference::LongTerm => VA_PICTURE_H264_LONG_TERM_REFERENCE,
            IsReference::ShortTerm => VA_PICTURE_H264_SHORT_TERM_REFERENCE,
        };

        PictureH264::new(surface.surface_id(), meta.frame_num, flags, meta.poc, meta.poc)
    }

    /// Builds [`BufferType::EncSequenceParameter`] from `sps`
    fn build_enc_seq_param(
        sps: &Sps,
        bits_per_second: u32,
        intra_period: u32,
        ip_period: u32,
    ) -> BufferType {
        let intra_idr_period = intra_period;

        let seq_fields = H264EncSeqFields::new(
            sps.chroma_format_idc as u32,
            sps.frame_mbs_only_flag as u32,
            sps.mb_adaptive_frame_field_flag as u32,
            sps.seq_scaling_matrix_present_flag as u32,
            sps.direct_8x8_inference_flag as u32,
            sps.log2_max_frame_num_minus4 as u32,
            sps.pic_order_cnt_type as u32,
            sps.log2_max_pic_order_cnt_lsb_minus4 as u32,
            sps.delta_pic_order_always_zero_flag as u32,
        );

        let frame_crop = if sps.frame_cropping_flag {
            Some(H264EncFrameCropOffsets::new(
                sps.frame_crop_left_offset,
                sps.frame_crop_right_offset,
                sps.frame_crop_top_offset,
                sps.frame_crop_bottom_offset,
            ))
        } else {
            None
        };

        let vui_fields = if sps.vui_parameters_present_flag {
            Some(H264VuiFields::new(
                sps.vui_parameters.aspect_ratio_idc as u32,
                sps.vui_parameters.timing_info_present_flag as u32,
                sps.vui_parameters.bitstream_restriction_flag as u32,
                sps.vui_parameters.log2_max_mv_length_horizontal,
                sps.vui_parameters.log2_max_mv_length_vertical,
                sps.vui_parameters.fixed_frame_rate_flag as u32,
                sps.vui_parameters.low_delay_hrd_flag as u32,
                sps.vui_parameters.motion_vectors_over_pic_boundaries_flag as u32,
            ))
        } else {
            None
        };

        let mut offset_for_ref_frame = [0i32; 256];
        offset_for_ref_frame[..255].copy_from_slice(&sps.offset_for_ref_frame[..]);

        BufferType::EncSequenceParameter(EncSequenceParameter::H264(
            EncSequenceParameterBufferH264::new(
                sps.seq_parameter_set_id,
                sps.level_idc as u8,
                intra_period,
                intra_idr_period,
                ip_period,
                bits_per_second,
                sps.max_num_ref_frames as u32,
                sps.pic_width_in_mbs_minus1 + 1,
                sps.pic_height_in_map_units_minus1 + 1,
                &seq_fields,
                sps.bit_depth_luma_minus8,
                sps.bit_depth_chroma_minus8,
                sps.num_ref_frames_in_pic_order_cnt_cycle,
                sps.offset_for_non_ref_pic,
                sps.offset_for_top_to_bottom_field,
                offset_for_ref_frame,
                frame_crop,
                vui_fields,
                sps.vui_parameters.aspect_ratio_idc,
                sps.vui_parameters.sar_width as u32,
                sps.vui_parameters.sar_height as u32,
                sps.vui_parameters.num_units_in_tick,
                sps.vui_parameters.time_scale,
            ),
        ))
    }

    /// Builds [`BufferType::EncPictureParameter`] from [`Request`] and sets bitstream
    /// output to `coded_buf`.
    fn build_enc_pic_param(
        request: &Request<'_, H>,
        coded_buf: &EncCodedBuffer,
        recon: &Reconstructed,
    ) -> BufferType {
        let pic_fields = H264EncPicFields::new(
            request.is_idr as u32,
            (request.dpb_meta.is_reference != IsReference::No) as u32,
            request.pps.entropy_coding_mode_flag as u32,
            request.pps.weighted_pred_flag as u32,
            request.pps.weighted_bipred_idc as u32,
            request.pps.constrained_intra_pred_flag as u32,
            request.pps.transform_8x8_mode_flag as u32,
            request.pps.deblocking_filter_control_present_flag as u32,
            request.pps.redundant_pic_cnt_present_flag as u32,
            0,
            request.pps.pic_scaling_matrix_present_flag as u32,
        );

        let curr_pic = Self::build_h264_pic(recon, &request.dpb_meta);

        assert!(request.ref_list_0.len() + request.ref_list_1.len() <= 16);

        let mut reference_frames: [PictureH264; 16] =
            std::array::from_fn(|_| Self::build_invalid_va_h264_pic_enc());

        for (idx, ref_frame) in
            request.ref_list_0.iter().chain(request.ref_list_1.iter()).enumerate().take(16)
        {
            reference_frames[idx] = Self::build_h264_pic(&ref_frame.recon_pic, &ref_frame.meta);
        }

        BufferType::EncPictureParameter(EncPictureParameter::H264(
            EncPictureParameterBufferH264::new(
                curr_pic,
                reference_frames,
                coded_buf.id(),
                request.pps.pic_parameter_set_id,
                request.pps.seq_parameter_set_id,
                0, // last_pic, don't appned EOS
                // The `frame_num` syntax element, which `DpbEntryMeta::frame_num` is.
                request.dpb_meta.frame_num as u16,
                (request.pps.pic_init_qp_minus26 + 26) as u8,
                request.pps.num_ref_idx_l0_default_active_minus1,
                request.pps.num_ref_idx_l1_default_active_minus1,
                request.pps.chroma_qp_index_offset,
                request.pps.second_chroma_qp_index_offset,
                &pic_fields,
            ),
        ))
    }

    /// Builds [`BufferType::EncSliceParameter`]
    fn build_enc_slice_param(
        pps: &Pps,
        header: &SliceHeader,
        ref_list_0: &[Rc<DpbEntry<Reconstructed>>],
        ref_list_1: &[Rc<DpbEntry<Reconstructed>>],
        num_macroblocks: u32,
    ) -> BufferType {
        let mut ref_pic_list_0: [PictureH264; 32] =
            std::array::from_fn(|_| Self::build_invalid_va_h264_pic_enc());

        for (idx, ref_frame) in ref_list_0.iter().enumerate().take(16) {
            ref_pic_list_0[idx] = Self::build_h264_pic(&ref_frame.recon_pic, &ref_frame.meta);
        }

        let mut ref_pic_list_1: [PictureH264; 32] =
            std::array::from_fn(|_| Self::build_invalid_va_h264_pic_enc());

        for (idx, ref_frame) in ref_list_1.iter().enumerate().take(16) {
            ref_pic_list_1[idx] = Self::build_h264_pic(&ref_frame.recon_pic, &ref_frame.meta);
        }

        let mut luma_weight_l0_flag = false;
        let mut luma_offset_l0 = [0i16; 32];

        if header.pred_weight_table.luma_weight_l0 != [0i16; 32] {
            luma_weight_l0_flag = true;
            for (i, val) in header.pred_weight_table.luma_offset_l0.iter().enumerate() {
                luma_offset_l0[i] = (*val).into();
            }
        }

        let mut chroma_weight_l0_flag = false;
        let mut chroma_offset_l0 = [[0i16; 2]; 32];

        if header.pred_weight_table.chroma_weight_l0 != [[0i16; 2]; 32] {
            chroma_weight_l0_flag = true;
            for (i, val) in header.pred_weight_table.chroma_offset_l0.iter().enumerate() {
                chroma_offset_l0[i] = [val[0].into(), val[1].into()];
            }
        }

        let mut luma_weight_l1_flag = false;
        let mut luma_offset_l1 = [0i16; 32];

        if header.pred_weight_table.luma_weight_l1 != [0i16; 32] {
            luma_weight_l1_flag = true;
            for (i, val) in header.pred_weight_table.luma_offset_l1.iter().enumerate() {
                luma_offset_l1[i] = *val;
            }
        }

        let mut chroma_weight_l1_flag = false;
        let mut chroma_offset_l1 = [[0i16; 2]; 32];

        if header.pred_weight_table.chroma_weight_l1 != [[0i16; 2]; 32] {
            chroma_weight_l1_flag = true;
            for (i, val) in header.pred_weight_table.chroma_offset_l1.iter().enumerate() {
                chroma_offset_l1[i] = [val[0].into(), val[1].into()];
            }
        }

        let (num_ref_idx_l0_active_minus1, num_ref_idx_l1_active_minus1) =
            if header.num_ref_idx_active_override_flag {
                (header.num_ref_idx_l0_active_minus1, header.num_ref_idx_l1_active_minus1)
            } else {
                (pps.num_ref_idx_l0_default_active_minus1, pps.num_ref_idx_l1_default_active_minus1)
            };
        BufferType::EncSliceParameter(EncSliceParameter::H264(EncSliceParameterBufferH264::new(
            header.first_mb_in_slice,
            num_macroblocks,
            VA_INVALID_ID,
            header.slice_type as u8,
            pps.pic_parameter_set_id,
            header.idr_pic_id,
            header.pic_order_cnt_lsb,
            header.delta_pic_order_cnt_bottom,
            header.delta_pic_order_cnt,
            header.direct_spatial_mv_pred_flag as u8,
            header.num_ref_idx_active_override_flag as u8,
            num_ref_idx_l0_active_minus1,
            num_ref_idx_l1_active_minus1,
            ref_pic_list_0,
            ref_pic_list_1,
            header.pred_weight_table.luma_log2_weight_denom,
            header.pred_weight_table.chroma_log2_weight_denom,
            luma_weight_l0_flag as u8,
            header.pred_weight_table.luma_weight_l0,
            luma_offset_l0,
            chroma_weight_l0_flag as u8,
            header.pred_weight_table.chroma_weight_l0,
            chroma_offset_l0,
            luma_weight_l1_flag as u8,
            header.pred_weight_table.luma_weight_l1,
            luma_offset_l1,
            chroma_weight_l1_flag as u8,
            header.pred_weight_table.chroma_weight_l1,
            chroma_offset_l1,
            header.cabac_init_idc,
            header.slice_qp_delta,
            header.disable_deblocking_filter_idc,
            header.slice_alpha_c0_offset_div2,
            header.slice_beta_offset_div2,
        )))
    }
}

impl<M, H> StatelessH264EncoderBackend for VaapiBackend<M, H>
where
    M: SurfaceMemoryDescriptor,
    H: Borrow<Surface<M>> + 'static,
{
    fn encode_slice(
        &mut self,
        request: Request<'_, H>,
    ) -> StatelessBackendResult<(Self::ReconPromise, Self::CodedPromise)> {
        let coded_buf = self.new_coded_buffer(&request.tunings.rate_control)?;
        let recon = self.new_scratch_picture()?;

        let pic_param = Self::build_enc_pic_param(&request, &coded_buf, &recon);
        let (nal_ref_idc, nalu_type) = request.slice_nal_header();
        let slice_param = Self::build_enc_slice_param(
            &request.pps,
            &request.header,
            &request.ref_list_0,
            &request.ref_list_1,
            request.num_macroblocks as u32,
        );

        // Clone reference frames
        let references: Vec<Rc<dyn Any>> = request
            .ref_list_0
            .iter()
            .cloned()
            .chain(request.ref_list_1.iter().cloned())
            .map(|entry| entry as Rc<dyn Any>)
            .collect();

        // Clone picture using [`Picture::new_from_same_surface`] to avoid
        // creatig a shared cell picture between its references and processed
        // picture.
        let mut picture =
            Picture::new(request.input_meta.timestamp, Rc::clone(self.context()), request.input);

        let rc_param =
            tunings_to_libva_rc::<{ MIN_QP as u32 }, { MAX_QP as u32 }>(&request.tunings)?;
        let rc_param = BufferType::EncMiscParameter(libva::EncMiscParameter::RateControl(rc_param));

        let framerate_param = BufferType::EncMiscParameter(libva::EncMiscParameter::FrameRate(
            libva::EncMiscParameterFrameRate::new(request.tunings.framerate, 0),
        ));

        // The sequence parameter buffer restates sequence-level state the driver keeps (the GOP
        // structure, the bit rate, the SPS fields and its timing), so it goes with the SPS, on
        // the IDR alone, as FFmpeg's `vaapi_encode.c` sends it. No driver takes it as the start
        // of a sequence (Mesa copies its fields; iHD reruns its sequence setup, without an IDR),
        // and none needs it between IDRs: a bit rate or frame rate a retune changes reaches the
        // driver through the rate-control and frame-rate buffers every frame carries.
        if request.parameter_sets.sps() {
            // Use bitrate from RateControl or ask driver to ignore
            let bits_per_second = request.tunings.rate_control.bitrate_target().unwrap_or(0) as u32;
            let seq_param = Self::build_enc_seq_param(
                &request.sps,
                bits_per_second,
                request.intra_period,
                request.ip_period,
            );
            picture.add_buffer(self.context().create_buffer(seq_param)?);
        }
        picture.add_buffer(self.context().create_buffer(pic_param)?);
        picture.add_buffer(self.context().create_buffer(rc_param)?);
        picture.add_buffer(self.context().create_buffer(framerate_param)?);

        let packed = self.packed_headers() != libva::VA_ENC_PACKED_HEADER_NONE;
        if packed {
            debug_assert!(
                request.coded_output.is_empty(),
                "packed-header backend: EncoderConfig::emit_parameter_sets must be false"
            );
            // The SPS and PPS end byte-aligned (`rbsp_trailing_bits()`); the slice header ends
            // where the slice data begins, mid-byte.
            if request.parameter_sets.sps() {
                self.add_packed_header(&mut picture, EncPackedHeaderType::Sequence, |w| {
                    Synthesizer::<Sps, _>::synthesize(3, &request.sps, w, true).map(|()| 0)
                })?;
            }
            if request.parameter_sets.pps() {
                self.add_packed_header(&mut picture, EncPackedHeaderType::Picture, |w| {
                    Synthesizer::<Pps, _>::synthesize(3, &request.pps, w, true).map(|()| 0)
                })?;
            }
            self.add_packed_header(&mut picture, EncPackedHeaderType::Slice, |w| {
                Synthesizer::<SliceHeader, _>::synthesize(
                    nal_ref_idc,
                    nalu_type,
                    &request.header,
                    &request.sps,
                    &request.pps,
                    w,
                    true,
                )
            })?;
        }
        picture.add_buffer(self.context().create_buffer(slice_param)?);

        // Start processing the picture encoding
        let picture = picture.begin().context("picture begin")?;
        let picture = picture.render().context("picture render")?;
        let picture = picture.end().context("picture end")?;

        // The coded buffer holds what the driver wrote: with packed headers the whole access
        // unit, from the SPS and PPS (where carried) to the slice. Without them it holds the
        // slice alone, behind the predictor's in-band SPS and PPS; a zero byte ahead of it keeps
        // the access unit's first NAL unit on a 4-byte start code (the zero_byte of H.264 B.1.2)
        // whichever start code the driver writes.
        let mut coded_output = request.coded_output;
        if !packed {
            coded_output.push(0);
        }

        // libva will handle the synchronization of reconstructed surface with implicit fences.
        // Therefore return the reconstructed frame immediately.
        let reference_promise = ReadyPromise::from(recon);

        let bitstream_promise =
            CodedOutputPromise::new(picture, references, coded_buf, coded_output);

        Ok((reference_promise, bitstream_promise))
    }
}

impl<V: VideoFrame> StatelessEncoder<V, VaapiBackend<V::MemDescriptor, Surface<V::MemDescriptor>>> {
    pub fn new_vaapi(
        display: Rc<Display>,
        config: EncoderConfig,
        fourcc: Fourcc,
        coded_size: Resolution,
        low_power: bool,
        blocking_mode: BlockingMode,
    ) -> EncodeResult<Self> {
        let va_profile = match config.profile {
            Profile::Baseline => VAProfile::VAProfileH264ConstrainedBaseline,
            Profile::Main => VAProfile::VAProfileH264Main,
            Profile::High => VAProfile::VAProfileH264High,
            _ => return Err(StatelessBackendError::UnsupportedProfile.into()),
        };

        let bitrate_control = rate_control_to_va_rc(&config.initial_tunings.rate_control)?;

        let entrypoint = if low_power { VAEntrypointEncSliceLP } else { VAEntrypointEncSlice };
        let packed_headers = query_packed_headers(&display, va_profile, entrypoint);

        let backend = VaapiBackend::new_with_packed_headers(
            display,
            va_profile,
            fourcc,
            coded_size,
            bitrate_control,
            low_power,
            packed_headers,
        )?;

        // Who writes the parameter sets is this backend's to decide (module doc): the driver,
        // from the packed SPS and PPS, where it takes packed headers; the predictor, in band,
        // where it does not.
        let mut config = config;
        config.emit_parameter_sets = packed_headers == libva::VA_ENC_PACKED_HEADER_NONE;

        Self::new_h264(backend, config, blocking_mode)
    }
}

#[cfg(test)]
#[path = "vaapi_packed_header_tests.rs"]
mod packed_header_tests;

#[cfg(test)]
pub(super) mod tests {
    use libva::Display;
    use libva::UsageHint;
    use libva::VAEntrypoint::VAEntrypointEncSliceLP;
    use libva::VAProfile::VAProfileH264Main;
    use libva::VA_RT_FORMAT_YUV420;

    use super::*;
    use crate::backend::vaapi::encoder::tests::upload_test_frame_nv12;
    use crate::backend::vaapi::encoder::tests::TestFrameGenerator;
    use crate::backend::vaapi::surface_pool::PooledVaSurface;
    use crate::backend::vaapi::surface_pool::VaSurfacePool;
    use crate::codec::h264::parser::Level;
    use crate::codec::h264::parser::PpsBuilder;
    use crate::codec::h264::parser::Profile;
    use crate::codec::h264::parser::SliceHeaderBuilder;
    use crate::codec::h264::parser::SliceType;
    use crate::codec::h264::parser::SpsBuilder;
    use crate::decoder::FramePool;
    use crate::encoder::simple_encode_loop;
    use crate::encoder::stateless::h264::BackendRequest;
    use crate::encoder::stateless::h264::EncoderConfig;
    use crate::encoder::stateless::h264::ParameterSets;
    use crate::encoder::stateless::h264::StatelessEncoder;
    use crate::encoder::stateless::BackendPromise;
    use crate::encoder::stateless::StatelessEncoderBackendImport;
    use crate::encoder::FrameMetadata;
    use crate::encoder::Tunings;
    use crate::FrameLayout;
    use crate::PlaneLayout;
    use crate::Resolution;

    #[test]
    // Ignore this test by default as it requires libva-compatible hardware.
    #[ignore]
    fn test_simple_encode_slice() {
        type Descriptor = ();
        type Surface = libva::Surface<Descriptor>;
        const WIDTH: u32 = 256;
        const HEIGHT: u32 = 256;
        let fourcc = b"NV12".into();

        let frame_layout = FrameLayout {
            format: (fourcc, 0),
            size: Resolution { width: WIDTH, height: HEIGHT },
            planes: vec![
                PlaneLayout { buffer_index: 0, offset: 0, stride: WIDTH as usize },
                PlaneLayout {
                    buffer_index: 0,
                    offset: (WIDTH * HEIGHT) as usize,
                    stride: WIDTH as usize,
                },
            ],
        };

        let display = Display::open().unwrap();
        let entrypoints = display.query_config_entrypoints(VAProfileH264Main).unwrap();
        let low_power = entrypoints.contains(&VAEntrypointEncSliceLP);

        let mut backend = VaapiBackend::<Descriptor, Surface>::new(
            Rc::clone(&display),
            VAProfileH264Main,
            fourcc,
            Resolution { width: WIDTH, height: HEIGHT },
            libva::VA_RC_CBR,
            low_power,
        )
        .unwrap();

        let mut surfaces = display
            .create_surfaces(
                VA_RT_FORMAT_YUV420,
                Some(frame_layout.format.0 .0),
                WIDTH,
                HEIGHT,
                Some(UsageHint::USAGE_HINT_ENCODER),
                vec![()],
            )
            .unwrap();

        let surface = surfaces.pop().unwrap();

        upload_test_frame_nv12(&display, &surface, 0.0);

        let input_meta =
            FrameMetadata { layout: frame_layout, force_keyframe: false, timestamp: 0 };

        let pic = backend.import_picture(&input_meta, surface).unwrap();

        let sps = SpsBuilder::new()
            .seq_parameter_set_id(0)
            .profile_idc(Profile::Main)
            .level_idc(Level::L4)
            .resolution(WIDTH, HEIGHT)
            .chroma_format_idc(3)
            .frame_mbs_only_flag(true)
            .direct_8x8_inference_flag(true)
            .max_num_ref_frames(1)
            .max_frame_num(32)
            .pic_order_cnt_type(0)
            .max_pic_order_cnt_lsb(128)
            .delta_pic_order_always_zero_flag(false)
            .bit_depth_chroma(8)
            .bit_depth_luma(8)
            .sar_resolution(1, 1)
            .build();

        let pps = PpsBuilder::new(Rc::clone(&sps))
            .pic_parameter_set_id(0)
            .pic_init_qp_minus26(0)
            .deblocking_filter_control_present_flag(true)
            .build();

        let header = SliceHeaderBuilder::new(&pps)
            .slice_type(SliceType::I)
            .first_mb_in_slice(0)
            .idr_pic_id(0)
            .build();

        let dpb_entry_meta =
            DpbEntryMeta { poc: 0, frame_num: 0, is_reference: IsReference::ShortTerm };

        let request = BackendRequest {
            sps: Rc::clone(&sps),
            pps: Rc::clone(&pps),
            header,
            dpb_meta: dpb_entry_meta,
            input: pic,
            input_meta,
            ref_list_0: vec![],
            ref_list_1: vec![],
            intra_period: 1,
            ip_period: 0,
            num_macroblocks: (WIDTH * HEIGHT) as usize / (16 * 16),
            is_idr: true,
            parameter_sets: ParameterSets::SpsAndPps,
            tunings: Tunings {
                rate_control: RateControl::ConstantBitrate(30_000),
                ..Default::default()
            },
            coded_output: vec![],
        };

        let (_, output) = backend.encode_slice(request).unwrap();
        let output = output.sync().unwrap();

        let write_to_file = std::option_env!("CROS_CODECS_TEST_WRITE_TO_FILE") == Some("true");
        if write_to_file {
            use std::io::Write;

            use crate::codec::h264::synthesizer::Synthesizer;
            let mut out = std::fs::File::create("test_simple_encode_slice.h264").unwrap();

            Synthesizer::<'_, Sps, &mut std::fs::File>::synthesize(3, &sps, &mut out, true)
                .unwrap();
            Synthesizer::<'_, Pps, &mut std::fs::File>::synthesize(3, &pps, &mut out, true)
                .unwrap();
            out.write_all(&output).unwrap();
            out.flush().unwrap();
        }
    }

    #[test]
    // Ignore this test by default as it requires libva-compatible hardware.
    #[ignore]
    fn test_vaapi_encoder() {
        type VaapiH264Encoder<'l> =
            StatelessEncoder<PooledVaSurface<()>, VaapiBackend<(), PooledVaSurface<()>>>;

        const WIDTH: usize = 512;
        const HEIGHT: usize = 512;

        let _ = env_logger::try_init();

        let display = libva::Display::open().unwrap();
        let entrypoints = display.query_config_entrypoints(VAProfileH264Main).unwrap();
        let low_power = entrypoints.contains(&VAEntrypointEncSliceLP);

        let config = EncoderConfig {
            profile: Profile::Main,
            resolution: Resolution { width: WIDTH as u32, height: HEIGHT as u32 },
            initial_tunings: Tunings {
                rate_control: RateControl::ConstantBitrate(1_200_000),
                framerate: 30,
                ..Default::default()
            },
            ..Default::default()
        };

        let frame_layout = FrameLayout {
            format: (b"NV12".into(), 0),
            size: Resolution { width: WIDTH as u32, height: HEIGHT as u32 },
            planes: vec![
                PlaneLayout { buffer_index: 0, offset: 0, stride: WIDTH },
                PlaneLayout { buffer_index: 0, offset: WIDTH * HEIGHT, stride: WIDTH },
            ],
        };

        let backend = VaapiBackend::<(), PooledVaSurface<()>>::new(
            Rc::clone(&display),
            VAProfileH264Main,
            frame_layout.format.0,
            frame_layout.size,
            libva::VA_RC_CBR,
            low_power,
        )
        .unwrap();
        let mut encoder =
            VaapiH264Encoder::new_h264(backend, config, BlockingMode::Blocking).unwrap();

        let mut pool = VaSurfacePool::new(
            Rc::clone(&display),
            VA_RT_FORMAT_YUV420,
            Some(UsageHint::USAGE_HINT_ENCODER),
            Resolution { width: WIDTH as u32, height: HEIGHT as u32 },
        );

        pool.add_frames(vec![(); 16]).unwrap();

        let mut frame_producer = TestFrameGenerator::new(100, display, pool, frame_layout);

        let mut bitstream = Vec::new();

        simple_encode_loop(&mut encoder, &mut frame_producer, |coded| {
            bitstream.extend(coded.bitstream)
        })
        .unwrap();

        let write_to_file = std::option_env!("CROS_CODECS_TEST_WRITE_TO_FILE") == Some("true");
        if write_to_file {
            use std::io::Write;
            let mut out = std::fs::File::create("test_vaapi_encoder.h264").unwrap();
            out.write_all(&bitstream).unwrap();
            out.flush().unwrap();
        }
    }
}
