// Copyright 2024 The ChromiumOS Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

//! VA-API stateless HEVC encoder backend. Mirrors
//! [`crate::encoder::stateless::h264::vaapi`], with the HEVC sequence / picture /
//! slice parameter builders and the **application-packed-header** path HEVC needs.
//!
//! ## Packed headers
//!
//! Some drivers require the application to supply the VPS / SPS / PPS and,
//! when advertised, the slice header as packed `VAEncPackedHeader*` buffers;
//! others self-generate them and advertise `VAConfigAttribEncPackedHeaders` as
//! `VA_ATTRIB_NOT_SUPPORTED`. At construction, [`query_packed_headers`] reads
//! the attribute and applies an all-three-or-none rule
//! ([`decide_packed_headers`]): if the driver advertises
//! all of SEQUENCE | PICTURE | SLICE, the backend packs all of them (VPS + SPS on
//! IDR, PPS every frame, and a slice header every frame) with the synthesizer;
//! otherwise it packs nothing and the driver self-generates. Either way the coded
//! bitstream lands wholly in the driver's coded buffer (`coded_output` stays
//! empty), producing a complete access unit.
//!
//! The three `build_enc_*_param` builders are free functions (not inherent
//! methods on [`VaapiBackend`]) because the H.264 backend already defines
//! identically-named methods on the same type.

use std::any::Any;
use std::borrow::Borrow;
use std::rc::Rc;

use anyhow::anyhow;
use anyhow::Context;
use libva::BufferType;
use libva::Display;
use libva::EncCodedBuffer;
use libva::EncPackedHeaderParameter;
use libva::EncPackedHeaderType;
use libva::EncPictureParameter;
use libva::EncPictureParameterBufferHEVC;
use libva::EncSequenceParameter;
use libva::EncSequenceParameterBufferHEVC;
use libva::EncSliceParameter;
use libva::EncSliceParameterBufferHEVC;
use libva::HEVCEncPicFields;
use libva::HEVCEncSeqFields;
use libva::HevcEncPicSccFields;
use libva::HevcEncSeqSccFields;
use libva::HevcEncSliceFields;
use libva::Picture;
use libva::PictureHEVC;
use libva::Surface;
use libva::SurfaceMemoryDescriptor;
use libva::VAEntrypoint::VAEntrypointEncSlice;
use libva::VAEntrypoint::VAEntrypointEncSliceLP;
use libva::VAProfile;
use libva::VA_INVALID_ID;
use libva::VA_PICTURE_HEVC_INVALID;

use crate::backend::vaapi::encoder::rate_control_to_va_rc;
use crate::backend::vaapi::encoder::tunings_to_libva_rc;
use crate::backend::vaapi::encoder::CodedOutputPromise;
use crate::backend::vaapi::encoder::Reconstructed;
use crate::backend::vaapi::encoder::VaapiBackend;
use crate::codec::h265::parser::Pps;
use crate::codec::h265::parser::Profile;
use crate::codec::h265::parser::SliceHeader;
use crate::codec::h265::parser::SliceType;
use crate::codec::h265::parser::Sps;
use crate::codec::h265::parser::Vps;
use crate::codec::h265::synthesizer::Synthesizer;
use crate::codec::h265::synthesizer::SynthesizerResult;
use crate::encoder::h265::EncoderConfig;
use crate::encoder::h265::H265;
use crate::encoder::stateless::h265::predictor::MAX_QP;
use crate::encoder::stateless::h265::predictor::MIN_QP;
use crate::encoder::stateless::h265::BackendRequest;
use crate::encoder::stateless::h265::DpbEntry;
use crate::encoder::stateless::h265::StatelessEncoder;
use crate::encoder::stateless::h265::StatelessH265EncoderBackend;
use crate::encoder::stateless::ReadyPromise;
use crate::encoder::stateless::StatelessBackendError;
use crate::encoder::stateless::StatelessBackendResult;
use crate::encoder::stateless::StatelessVideoEncoderBackend;
use crate::encoder::EncodeResult;
use crate::video_frame::VideoFrame;
use crate::BlockingMode;
use crate::Fourcc;
use crate::Resolution;

type Request<H> = BackendRequest<H, Reconstructed>;

/// VA `coding_type`: I = 1, P = 2 (B = 3, unused here).
const CODING_TYPE_I: u32 = 1;
const CODING_TYPE_P: u32 = 2;

/// The VA `coding_type` for a slice. It must track whether the *slice* is intra
/// (`SliceType::I -> CODING_TYPE_I`), never `is_idr`: a forced keyframe is an I
/// slice and must be coded as `CODING_TYPE_I` regardless of the IDR flag, and a
/// P slice is always `CODING_TYPE_P`.
pub(crate) fn coding_type_for_slice(slice_type: SliceType) -> u32 {
    match slice_type {
        SliceType::I => CODING_TYPE_I,
        _ => CODING_TYPE_P,
    }
}

/// `collocated_ref_pic_index` sentinel when `slice_temporal_mvp_enabled_flag == 0`
/// Temporal MVP is disabled, so there is no collocated picture.
const COLLOCATED_REF_PIC_NONE: u8 = 0xFF;

/// The all-three-or-none packed-header decision. Given the queried
/// `VAConfigAttribEncPackedHeaders` value, returns the mask of headers the
/// application must supply: the full `SEQUENCE | PICTURE | SLICE` set when the
/// driver advertises all three, or `0` (`VA_ENC_PACKED_HEADER_NONE`) when
/// the attribute is unsupported or only partially
/// advertised. Pure — unit tested without a driver.
pub(crate) fn decide_packed_headers(attrib_value: u32) -> u32 {
    const NEED: u32 = libva::VA_ENC_PACKED_HEADER_SEQUENCE
        | libva::VA_ENC_PACKED_HEADER_PICTURE
        | libva::VA_ENC_PACKED_HEADER_SLICE;
    if attrib_value == libva::VA_ATTRIB_NOT_SUPPORTED {
        return libva::VA_ENC_PACKED_HEADER_NONE;
    }
    if attrib_value & NEED == NEED {
        NEED
    } else {
        libva::VA_ENC_PACKED_HEADER_NONE
    }
}

/// Query the driver's packed-header support for `profile`/`entrypoint` and apply
/// [`decide_packed_headers`]. A query failure is treated as "self-generating"
/// (`0`) for drivers that generate the headers themselves.
fn query_packed_headers(
    display: &Display,
    profile: VAProfile::Type,
    entrypoint: libva::VAEntrypoint::Type,
) -> u32 {
    let mut attrs = [libva::VAConfigAttrib {
        type_: libva::VAConfigAttribType::VAConfigAttribEncPackedHeaders,
        value: 0,
    }];
    match display.get_config_attributes(profile, entrypoint, &mut attrs) {
        Ok(()) => decide_packed_headers(attrs[0].value),
        Err(_) => libva::VA_ENC_PACKED_HEADER_NONE,
    }
}

/// An invalid [`PictureHEVC`] used to fill the unused reference-list slots.
fn invalid_hevc_pic() -> PictureHEVC {
    PictureHEVC::new(VA_INVALID_ID, 0, VA_PICTURE_HEVC_INVALID)
}

/// A reference / current [`PictureHEVC`] from a surface id + POC. `flags` are `0`
/// for references because the driver derives the RPS categories from the packed
/// slice header or the reference POCs.
fn hevc_pic(surface_id: u32, poc: i32) -> PictureHEVC {
    PictureHEVC::new(surface_id, poc, 0)
}

/// `[PictureHEVC; 15]` reference-frame array with `ref_list_0` filled and the
/// tail marked invalid.
fn reference_frames(ref_list_0: &[Rc<DpbEntry<Reconstructed>>]) -> [PictureHEVC; 15] {
    let mut frames: [PictureHEVC; 15] = std::array::from_fn(|_| invalid_hevc_pic());
    for (idx, ref_frame) in ref_list_0.iter().enumerate().take(15) {
        frames[idx] = hevc_pic(ref_frame.recon_pic.surface_id(), ref_frame.meta.poc);
    }
    frames
}

/// Builds [`BufferType::EncSequenceParameter`] from `sps`.
fn build_enc_seq_param(
    sps: &Sps,
    bits_per_second: u32,
    intra_period: u32,
    ip_period: u32,
) -> BufferType {
    let ptl = &sps.profile_tier_level;
    let seq_fields = HEVCEncSeqFields::new(
        sps.chroma_format_idc as u32,
        sps.separate_colour_plane_flag as u32,
        sps.bit_depth_luma_minus8 as u32,
        sps.bit_depth_chroma_minus8 as u32,
        sps.scaling_list_enabled_flag as u32,
        sps.strong_intra_smoothing_enabled_flag as u32,
        sps.amp_enabled_flag as u32,
        sps.sample_adaptive_offset_enabled_flag as u32,
        sps.pcm_enabled_flag as u32,
        sps.pcm_loop_filter_disabled_flag as u32,
        sps.temporal_mvp_enabled_flag as u32,
        /* low_delay_seq */ 1,
        /* hierachical_flag */ 0,
    );

    let scc_fields = HevcEncSeqSccFields::new(0);

    BufferType::EncSequenceParameter(EncSequenceParameter::HEVC(
        EncSequenceParameterBufferHEVC::new(
            ptl.general_profile_idc,
            ptl.general_level_idc as u8,
            ptl.general_tier_flag as u8,
            intra_period,
            intra_period, // intra_idr_period
            ip_period,
            bits_per_second,
            sps.pic_width_in_luma_samples,
            sps.pic_height_in_luma_samples,
            &seq_fields,
            sps.log2_min_luma_coding_block_size_minus3,
            sps.log2_diff_max_min_luma_coding_block_size,
            sps.log2_min_luma_transform_block_size_minus2,
            sps.log2_diff_max_min_luma_transform_block_size,
            sps.max_transform_hierarchy_depth_inter,
            sps.max_transform_hierarchy_depth_intra,
            0, // pcm_sample_bit_depth_luma_minus1
            0, // pcm_sample_bit_depth_chroma_minus1
            0, // log2_min_pcm_luma_coding_block_size_minus3
            0, // log2_max_pcm_luma_coding_block_size_minus3
            None,
            0, // aspect_ratio_idc
            0, // sar_width
            0, // sar_height
            0, // vui_num_units_in_tick
            0, // vui_time_scale
            0, // min_spatial_segmentation_idc
            0, // max_bytes_per_pic_denom
            0, // max_bits_per_min_cu_denom
            &scc_fields,
        ),
    ))
}

/// Builds [`BufferType::EncPictureParameter`] from the request, targeting
/// `coded_buf` and reconstructing into `recon`.
fn build_enc_pic_param<H>(
    request: &Request<H>,
    coded_buf: &EncCodedBuffer,
    recon: &Reconstructed,
) -> BufferType {
    let pps = &request.pps;
    // Derive coding_type from the slice type, never from is_idr: a forced-keyframe
    // I slice must be CODING_TYPE_I even though its ref list is empty, and this
    // stays consistent with the IDR NAL type + `is_idr` flag.
    let coding_type = coding_type_for_slice(request.header.type_);

    let pic_fields = HEVCEncPicFields::new(
        request.is_idr as u32,
        coding_type,
        /* reference_pic_flag: every LowDelay frame is a reference */ 1,
        pps.dependent_slice_segments_enabled_flag as u32,
        pps.sign_data_hiding_enabled_flag as u32,
        pps.constrained_intra_pred_flag as u32,
        pps.transform_skip_enabled_flag as u32,
        pps.cu_qp_delta_enabled_flag as u32,
        pps.weighted_pred_flag as u32,
        pps.weighted_bipred_flag as u32,
        pps.transquant_bypass_enabled_flag as u32,
        pps.tiles_enabled_flag as u32,
        pps.entropy_coding_sync_enabled_flag as u32,
        pps.loop_filter_across_tiles_enabled_flag as u32,
        pps.loop_filter_across_slices_enabled_flag as u32,
        pps.scaling_list_data_present_flag as u32,
        /* screen_content_flag */ 0,
        /* enable_gpu_weighted_prediction */ 0,
        request.header.no_output_of_prior_pics_flag as u32,
    );

    let curr_pic = hevc_pic(recon.surface_id(), request.dpb_meta.poc);
    let refs = reference_frames(&request.ref_list_0);
    let scc_fields = HevcEncPicSccFields::new(0);
    let nal_unit_type = request.nalu_type as u8;

    BufferType::EncPictureParameter(EncPictureParameter::HEVC(EncPictureParameterBufferHEVC::new(
        curr_pic,
        refs,
        coded_buf.id(),
        COLLOCATED_REF_PIC_NONE,
        0, // last_picture — never signal EOS/EOSEQ (the muxer owns framing)
        (pps.init_qp_minus26 + 26) as u8,
        pps.diff_cu_qp_delta_depth,
        pps.cb_qp_offset,
        pps.cr_qp_offset,
        pps.num_tile_columns_minus1,
        pps.num_tile_rows_minus1,
        [0u8; 19],
        [0u8; 21],
        pps.log2_parallel_merge_level_minus2,
        0, // ctu_max_bitsize_allowed
        pps.num_ref_idx_l0_default_active_minus1,
        pps.num_ref_idx_l1_default_active_minus1,
        pps.pic_parameter_set_id,
        nal_unit_type,
        &pic_fields,
        0, // hierarchical_level_plus1
        0, // va_byte_reserved
        &scc_fields,
    )))
}

/// Builds [`BufferType::EncSliceParameter`].
fn build_enc_slice_param(
    pps: &Pps,
    header: &SliceHeader,
    ref_list_0: &[Rc<DpbEntry<Reconstructed>>],
    num_ctu_in_slice: u32,
) -> BufferType {
    let ref_pic_list0 = reference_frames(ref_list_0);
    let ref_pic_list1: [PictureHEVC; 15] = std::array::from_fn(|_| invalid_hevc_pic());

    let (num_ref_idx_l0_active_minus1, num_ref_idx_l1_active_minus1) =
        if header.num_ref_idx_active_override_flag {
            (header.num_ref_idx_l0_active_minus1, header.num_ref_idx_l1_active_minus1)
        } else {
            (pps.num_ref_idx_l0_default_active_minus1, pps.num_ref_idx_l1_default_active_minus1)
        };

    let slice_fields = HevcEncSliceFields::new(
        /* last_slice_of_pic_flag */ 1,
        header.dependent_slice_segment_flag as u32,
        header.colour_plane_id as u32,
        header.temporal_mvp_enabled_flag as u32,
        header.sao_luma_flag as u32,
        header.sao_chroma_flag as u32,
        header.num_ref_idx_active_override_flag as u32,
        header.mvd_l1_zero_flag as u32,
        header.cabac_init_flag as u32,
        header.deblocking_filter_disabled_flag as u32,
        header.loop_filter_across_slices_enabled_flag as u32,
        header.collocated_from_l0_flag as u32,
    );

    BufferType::EncSliceParameter(EncSliceParameter::HEVC(EncSliceParameterBufferHEVC::new(
        /* slice_segment_address */ 0,
        num_ctu_in_slice,
        header.type_ as u8,
        pps.pic_parameter_set_id,
        num_ref_idx_l0_active_minus1,
        num_ref_idx_l1_active_minus1,
        ref_pic_list0,
        ref_pic_list1,
        header.pred_weight_table.luma_log2_weight_denom,
        header.pred_weight_table.delta_chroma_log2_weight_denom,
        [0i8; 15],
        [0i8; 15],
        [[0i8; 2]; 15],
        [[0i8; 2]; 15],
        [0i8; 15],
        [0i8; 15],
        [[0i8; 2]; 15],
        [[0i8; 2]; 15],
        5 - header.five_minus_max_num_merge_cand,
        header.qp_delta,
        header.cb_qp_offset,
        header.cr_qp_offset,
        header.beta_offset_div2,
        header.tc_offset_div2,
        &slice_fields,
        /* pred_weight_table_bit_offset */ 0,
        /* pred_weight_table_bit_length */ 0,
    )))
}

/// Run a synthesizer closure into a fresh byte buffer, mapping its error into a
/// backend error.
fn synth(f: impl FnOnce(&mut Vec<u8>) -> SynthesizerResult<()>) -> StatelessBackendResult<Vec<u8>> {
    let mut buf = Vec::new();
    f(&mut buf).map_err(|e| StatelessBackendError::Other(anyhow!("hevc packed header: {e}")))?;
    Ok(buf)
}

impl<M, H> StatelessVideoEncoderBackend<H265> for VaapiBackend<M, H>
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
    /// Synthesise a NAL into an application-packed header and queue the
    /// `VAEncPackedHeaderParameterBuffer` + `VAEncPackedHeaderDataBuffer` pair on
    /// `picture`. The synthesizer emits a complete, byte-aligned Annex-B NAL with
    /// emulation-prevention on, so `bit_length == bytes * 8` and
    /// `has_emulation_bytes == true`.
    fn add_packed_header(
        &self,
        picture: &mut Picture<libva::PictureNew, H>,
        header_type: EncPackedHeaderType,
        data: Vec<u8>,
    ) -> StatelessBackendResult<()> {
        let bit_length = (data.len() * 8) as u32;
        let param = BufferType::EncPackedHeaderParameter(EncPackedHeaderParameter::new(
            header_type,
            bit_length,
            /* has_emulation */ true,
        ));
        picture.add_buffer(self.context().create_buffer(param)?);
        picture.add_buffer(self.context().create_buffer(BufferType::EncPackedHeaderData(data))?);
        Ok(())
    }
}

impl<M, H> StatelessH265EncoderBackend for VaapiBackend<M, H>
where
    M: SurfaceMemoryDescriptor,
    H: Borrow<Surface<M>> + 'static,
{
    fn encode_slice(
        &mut self,
        request: Request<H>,
    ) -> StatelessBackendResult<(Self::ReconPromise, Self::CodedPromise)> {
        let coded_buf = self.new_coded_buffer(&request.tunings.rate_control)?;
        let recon = self.new_scratch_picture()?;

        let pic_param = build_enc_pic_param(&request, &coded_buf, &recon);
        let slice_param = build_enc_slice_param(
            &request.pps,
            &request.header,
            &request.ref_list_0,
            request.num_ctu_in_slice,
        );

        // Hold references alive while the picture is processed.
        let references: Vec<Rc<dyn Any>> =
            request.ref_list_0.iter().cloned().map(|entry| entry as Rc<dyn Any>).collect();

        let mut picture =
            Picture::new(request.input_meta.timestamp, Rc::clone(self.context()), request.input);

        let rc_param =
            tunings_to_libva_rc::<{ MIN_QP as u32 }, { MAX_QP as u32 }>(&request.tunings)?;
        let rc_param = BufferType::EncMiscParameter(libva::EncMiscParameter::RateControl(rc_param));
        let framerate_param = BufferType::EncMiscParameter(libva::EncMiscParameter::FrameRate(
            libva::EncMiscParameterFrameRate::new(request.tunings.framerate, 0),
        ));

        // The sequence parameter defines the coded video sequence; submit it only
        // on IDR (a new CVS), matching the already-IDR-gated packed VPS/SPS. A
        // mid-GOP seq param can make a driver
        // treat it as a sequence boundary.
        if request.is_idr {
            let bits_per_second =
                request.tunings.rate_control.bitrate_target().unwrap_or(0) as u32;
            let seq_param = build_enc_seq_param(
                &request.sps,
                bits_per_second,
                request.intra_period,
                request.ip_period,
            );
            picture.add_buffer(self.context().create_buffer(seq_param)?);
        }
        picture.add_buffer(self.context().create_buffer(rc_param)?);
        picture.add_buffer(self.context().create_buffer(framerate_param)?);

        // Application-packed headers: VPS + SPS on IDR, PPS every frame, then
        // the picture parameter buffer, then the packed slice header + slice
        // parameter buffer. When the driver self-generates (`packed_headers == 0`),
        // nothing is packed.
        let packed = self.packed_headers();
        if packed != 0 {
            if request.is_idr {
                let vps = synth(|w| Synthesizer::<Vps, _>::synthesize(request.vps.as_ref(), w, true))?;
                self.add_packed_header(&mut picture, EncPackedHeaderType::Sequence, vps)?;

                let sps = synth(|w| Synthesizer::<Sps, _>::synthesize(request.sps.as_ref(), w, true))?;
                self.add_packed_header(&mut picture, EncPackedHeaderType::Sequence, sps)?;
            }
            let pps = synth(|w| Synthesizer::<Pps, _>::synthesize(request.pps.as_ref(), w, true))?;
            self.add_packed_header(&mut picture, EncPackedHeaderType::Picture, pps)?;
        }

        picture.add_buffer(self.context().create_buffer(pic_param)?);

        if packed != 0 {
            let slice_hdr = synth(|w| {
                Synthesizer::<SliceHeader, _>::synthesize(
                    request.nalu_type,
                    &request.header,
                    &request.sps,
                    &request.pps,
                    w,
                    true,
                )
            })?;
            self.add_packed_header(&mut picture, EncPackedHeaderType::Slice, slice_hdr)?;
        }

        picture.add_buffer(self.context().create_buffer(slice_param)?);

        let picture = picture.begin().context("picture begin")?;
        let picture = picture.render().context("picture render")?;
        let picture = picture.end().context("picture end")?;

        // For HEVC the driver's coded buffer holds the whole access unit (packed
        // or self-generated parameter sets + slice); `coded_output` is empty.
        let coded_output = request.coded_output;

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
        // HEVC scope: Main (8-bit 4:2:0), Main10 (10-bit 4:2:0), and Main 4:2:2
        // 10 (RExt, 10-bit 4:2:2). The predictor derives the matching bit depth and
        // chroma format from the same profile; the fourcc (NV12 / P010 / Y210)
        // must agree (the caller pairs them). Any other profile is rejected
        // cleanly rather than silently mis-encoded.
        let va_profile = match config.profile {
            Profile::Main => VAProfile::VAProfileHEVCMain,
            Profile::Main10 => VAProfile::VAProfileHEVCMain10,
            Profile::RangeExtensions => VAProfile::VAProfileHEVCMain422_10,
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

        Self::new_h265(backend, config, blocking_mode)
    }
}
