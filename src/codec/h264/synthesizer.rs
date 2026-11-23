// Copyright 2024 The ChromiumOS Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
use std::fmt;
use std::io::Write;

use crate::codec::h264::nalu_writer::NaluWriter;
use crate::codec::h264::nalu_writer::NaluWriterError;
use crate::codec::h264::parser::HrdParams;
use crate::codec::h264::parser::NaluType;
use crate::codec::h264::parser::Pps;
use crate::codec::h264::parser::RefPicListModification;
use crate::codec::h264::parser::SliceHeader;
use crate::codec::h264::parser::SliceType;
use crate::codec::h264::parser::Sps;
use crate::codec::h264::parser::DEFAULT_4X4_INTER;
use crate::codec::h264::parser::DEFAULT_4X4_INTRA;
use crate::codec::h264::parser::DEFAULT_8X8_INTER;
use crate::codec::h264::parser::DEFAULT_8X8_INTRA;

#[cfg(test)]
#[path = "slice_header_synthesis_tests.rs"]
mod slice_header_synthesis_tests;

mod private {
    pub trait NaluStruct {}
}

impl private::NaluStruct for Sps {}

impl private::NaluStruct for Pps {}

impl private::NaluStruct for SliceHeader {}

#[derive(Debug)]
pub enum SynthesizerError {
    Unsupported,
    /// The values given break a constraint of the specification; names which.
    Invalid(&'static str),
    NaluWriter(NaluWriterError),
}

impl fmt::Display for SynthesizerError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            SynthesizerError::Unsupported => write!(f, "tried to synthesize unsupported settings"),
            SynthesizerError::Invalid(what) => {
                write!(f, "tried to synthesize an invalid header: {what}")
            }
            SynthesizerError::NaluWriter(x) => write!(f, "{}", x.to_string()),
        }
    }
}

impl From<NaluWriterError> for SynthesizerError {
    fn from(err: NaluWriterError) -> Self {
        SynthesizerError::NaluWriter(err)
    }
}

pub type SynthesizerResult<T> = Result<T, SynthesizerError>;

/// A helper to output typed NALUs to [`std::io::Write`] using [`NaluWriter`].
pub struct Synthesizer<'n, N: private::NaluStruct, W: Write> {
    writer: NaluWriter<W>,
    nalu: &'n N,
}

/// Extended Sample Aspect Ratio - H.264 Table E-1
const EXTENDED_SAR: u8 = 255;

impl<N: private::NaluStruct, W: Write> Synthesizer<'_, N, W> {
    fn u<T: Into<u32>>(&mut self, bits: usize, value: T) -> SynthesizerResult<()> {
        self.writer.write_u(bits, value)?;
        Ok(())
    }

    fn f<T: Into<u32>>(&mut self, bits: usize, value: T) -> SynthesizerResult<()> {
        self.writer.write_f(bits, value)?;
        Ok(())
    }

    fn ue<T: Into<u32>>(&mut self, value: T) -> SynthesizerResult<()> {
        self.writer.write_ue(value)?;
        Ok(())
    }

    fn se<T: Into<i32>>(&mut self, value: T) -> SynthesizerResult<()> {
        self.writer.write_se(value)?;
        Ok(())
    }

    fn scaling_list(&mut self, list: &[u8], default: &[u8]) -> SynthesizerResult<()> {
        // H.264 7.3.2.1.1.1
        if list == default {
            self.se(-8)?;
            return Ok(());
        }

        // The number of list values we want to encode.
        let mut run = list.len();

        // Check how many values at the end of the matrix are the same,
        // so we can save on encoding those.
        for j in (1..list.len()).rev() {
            if list[j - 1] != list[j] {
                break;
            }
            run -= 1;
        }

        // Encode deltas.
        let mut last_scale = 8;
        for scale in &list[0..run] {
            let delta_scale = *scale as i32 - last_scale;
            self.se(delta_scale)?;
            last_scale = *scale as i32;
        }

        // Didn't encode all values, encode -|last_scale| to set decoder's
        // |next_scale| (H.264 7.3.2.1.1.1) to zero, i.e. decoder should repeat
        // last values in matrix.
        if run < list.len() {
            self.se(-last_scale)?;
        }

        Ok(())
    }

    fn default_scaling_list(i: usize) -> &'static [u8] {
        // H.264 Table 7-2
        match i {
            0 => &DEFAULT_4X4_INTRA[..],
            1 => &DEFAULT_4X4_INTRA[..],
            2 => &DEFAULT_4X4_INTRA[..],
            3 => &DEFAULT_4X4_INTER[..],
            4 => &DEFAULT_4X4_INTER[..],
            5 => &DEFAULT_4X4_INTER[..],
            6 => &DEFAULT_8X8_INTRA[..],
            7 => &DEFAULT_8X8_INTER[..],
            8 => &DEFAULT_8X8_INTRA[..],
            9 => &DEFAULT_8X8_INTER[..],
            10 => &DEFAULT_8X8_INTRA[..],
            11 => &DEFAULT_8X8_INTER[..],
            _ => unreachable!(),
        }
    }

    fn rbsp_trailing_bits(&mut self) -> SynthesizerResult<()> {
        self.f(1, 1u32)?;

        while !self.writer.aligned() {
            self.f(1, 0u32)?;
        }

        Ok(())
    }
}

impl<'n, W: Write> Synthesizer<'n, Sps, W> {
    pub fn synthesize(
        ref_idc: u8,
        sps: &'n Sps,
        writer: W,
        ep_enabled: bool,
    ) -> SynthesizerResult<()> {
        let mut s = Self { writer: NaluWriter::<W>::new(writer, ep_enabled), nalu: sps };

        s.writer.write_header(ref_idc, NaluType::Sps as u8)?;
        s.seq_parameter_set_data()?;
        s.rbsp_trailing_bits()
    }

    fn hrd_parameters(&mut self, hrd_params: &HrdParams) -> SynthesizerResult<()> {
        self.ue(hrd_params.cpb_cnt_minus1)?;
        self.u(4, hrd_params.bit_rate_scale)?;
        self.u(4, hrd_params.cpb_size_scale)?;

        for i in 0..=(hrd_params.cpb_cnt_minus1 as usize) {
            self.ue(hrd_params.bit_rate_value_minus1[i])?;
            self.ue(hrd_params.cpb_size_value_minus1[i])?;
            self.u(1, hrd_params.cbr_flag[i])?;
        }

        self.u(5, hrd_params.initial_cpb_removal_delay_length_minus1)?;
        self.u(5, hrd_params.cpb_removal_delay_length_minus1)?;
        self.u(5, hrd_params.dpb_output_delay_length_minus1)?;
        self.u(5, hrd_params.time_offset_length)?;

        Ok(())
    }

    fn vui_parameters(&mut self) -> SynthesizerResult<()> {
        // H.264 E.1.1
        let vui_params = &self.nalu.vui_parameters;

        self.u(1, vui_params.aspect_ratio_info_present_flag)?;
        if vui_params.aspect_ratio_info_present_flag {
            self.u(8, vui_params.aspect_ratio_idc)?;
            if vui_params.aspect_ratio_idc == EXTENDED_SAR {
                self.u(16, vui_params.sar_width)?;
                self.u(16, vui_params.sar_height)?;
            }
        }

        self.u(1, vui_params.overscan_info_present_flag)?;
        if vui_params.overscan_info_present_flag {
            self.u(1, vui_params.overscan_appropriate_flag)?;
        }

        self.u(1, vui_params.video_signal_type_present_flag)?;
        if vui_params.video_signal_type_present_flag {
            self.u(3, vui_params.video_format)?;
            self.u(1, vui_params.video_full_range_flag)?;

            self.u(1, vui_params.colour_description_present_flag)?;
            if vui_params.colour_description_present_flag {
                self.u(8, vui_params.colour_primaries)?;
                self.u(8, vui_params.transfer_characteristics)?;
                self.u(8, vui_params.matrix_coefficients)?;
            }
        }

        self.u(1, vui_params.chroma_loc_info_present_flag)?;
        if vui_params.chroma_loc_info_present_flag {
            self.ue(vui_params.chroma_sample_loc_type_top_field)?;
            self.ue(self.nalu.vui_parameters.chroma_sample_loc_type_bottom_field)?;
        }

        self.u(1, vui_params.timing_info_present_flag)?;
        if vui_params.timing_info_present_flag {
            self.u(32, vui_params.num_units_in_tick)?;
            self.u(32, vui_params.time_scale)?;
            self.u(1, vui_params.fixed_frame_rate_flag)?;
        }

        self.u(1, vui_params.nal_hrd_parameters_present_flag)?;
        if vui_params.nal_hrd_parameters_present_flag {
            self.hrd_parameters(&vui_params.nal_hrd_parameters)?;
        }
        self.u(1, vui_params.vcl_hrd_parameters_present_flag)?;
        if vui_params.vcl_hrd_parameters_present_flag {
            self.hrd_parameters(&vui_params.vcl_hrd_parameters)?;
        }

        if vui_params.nal_hrd_parameters_present_flag || vui_params.vcl_hrd_parameters_present_flag
        {
            self.u(1, vui_params.low_delay_hrd_flag)?;
        }

        self.u(1, vui_params.pic_struct_present_flag)?;

        self.u(1, vui_params.bitstream_restriction_flag)?;
        if vui_params.bitstream_restriction_flag {
            self.u(1, vui_params.motion_vectors_over_pic_boundaries_flag)?;
            self.ue(vui_params.max_bytes_per_pic_denom)?;
            self.ue(vui_params.max_bits_per_mb_denom)?;
            self.ue(vui_params.log2_max_mv_length_horizontal)?;
            self.ue(vui_params.log2_max_mv_length_vertical)?;
            self.ue(vui_params.max_num_reorder_frames)?;
            self.ue(vui_params.max_dec_frame_buffering)?;
        }

        Ok(())
    }

    fn seq_parameter_set_data(&mut self) -> SynthesizerResult<()> {
        // H.264 7.3.2.1.1
        self.u(8, self.nalu.profile_idc)?;
        self.u(1, self.nalu.constraint_set0_flag)?;
        self.u(1, self.nalu.constraint_set1_flag)?;
        self.u(1, self.nalu.constraint_set2_flag)?;
        self.u(1, self.nalu.constraint_set3_flag)?;
        self.u(1, self.nalu.constraint_set4_flag)?;
        self.u(1, self.nalu.constraint_set5_flag)?;
        self.u(2, /* reserved_zero_2bits */ 0u32)?;
        self.u(8, self.nalu.level_idc as u32)?;
        self.ue(self.nalu.seq_parameter_set_id)?;

        if self.nalu.profile_idc == 100
            || self.nalu.profile_idc == 110
            || self.nalu.profile_idc == 122
            || self.nalu.profile_idc == 244
            || self.nalu.profile_idc == 44
            || self.nalu.profile_idc == 83
            || self.nalu.profile_idc == 86
            || self.nalu.profile_idc == 118
            || self.nalu.profile_idc == 128
            || self.nalu.profile_idc == 138
            || self.nalu.profile_idc == 139
            || self.nalu.profile_idc == 134
            || self.nalu.profile_idc == 135
        {
            self.ue(self.nalu.chroma_format_idc)?;

            if self.nalu.chroma_format_idc == 3 {
                self.u(1, self.nalu.separate_colour_plane_flag)?;
            }

            self.ue(self.nalu.bit_depth_luma_minus8)?;
            self.ue(self.nalu.bit_depth_chroma_minus8)?;
            self.u(1, self.nalu.qpprime_y_zero_transform_bypass_flag)?;
            self.u(1, self.nalu.seq_scaling_matrix_present_flag)?;

            if self.nalu.seq_scaling_matrix_present_flag {
                let scaling_list_count = if self.nalu.chroma_format_idc != 3 { 8 } else { 12 };

                for i in 0..scaling_list_count {
                    // Assume if scaling lists are zeroed that they are not present.
                    if i < 6 {
                        if self.nalu.scaling_lists_4x4[i] == [0; 16] {
                            self.u(1, /* seq_scaling_list_present_flag */ false)?;
                        } else {
                            self.u(1, /* seq_scaling_list_present_flag */ true)?;
                            self.scaling_list(
                                &self.nalu.scaling_lists_4x4[i],
                                Self::default_scaling_list(i),
                            )?;
                        }
                    } else if self.nalu.scaling_lists_8x8[i - 6] == [0; 64] {
                        self.u(1, /* seq_scaling_list_present_flag */ false)?;
                    } else {
                        self.u(1, /* seq_scaling_list_present_flag */ true)?;
                        self.scaling_list(
                            &self.nalu.scaling_lists_8x8[i - 6],
                            Self::default_scaling_list(i),
                        )?;
                    }
                }
            }
        }

        self.ue(self.nalu.log2_max_frame_num_minus4)?;
        self.ue(self.nalu.pic_order_cnt_type)?;

        if self.nalu.pic_order_cnt_type == 0 {
            self.ue(self.nalu.log2_max_pic_order_cnt_lsb_minus4)?;
        } else if self.nalu.pic_order_cnt_type == 1 {
            self.u(1, self.nalu.delta_pic_order_always_zero_flag)?;
            self.se(self.nalu.offset_for_non_ref_pic)?;
            self.se(self.nalu.offset_for_top_to_bottom_field)?;
            self.ue(self.nalu.num_ref_frames_in_pic_order_cnt_cycle)?;

            for offset_for_ref_frame in &self.nalu.offset_for_ref_frame {
                self.se(*offset_for_ref_frame)?;
            }
        }

        self.ue(self.nalu.max_num_ref_frames)?;
        self.u(1, self.nalu.gaps_in_frame_num_value_allowed_flag)?;
        self.ue(self.nalu.pic_width_in_mbs_minus1)?;
        self.ue(self.nalu.pic_height_in_map_units_minus1)?;
        self.u(1, self.nalu.frame_mbs_only_flag)?;
        if !self.nalu.frame_mbs_only_flag {
            self.u(1, self.nalu.mb_adaptive_frame_field_flag)?;
        }
        self.u(1, self.nalu.direct_8x8_inference_flag)?;

        self.u(1, self.nalu.frame_cropping_flag)?;
        if self.nalu.frame_cropping_flag {
            self.ue(self.nalu.frame_crop_left_offset)?;
            self.ue(self.nalu.frame_crop_right_offset)?;
            self.ue(self.nalu.frame_crop_top_offset)?;
            self.ue(self.nalu.frame_crop_bottom_offset)?;
        }

        self.u(1, self.nalu.vui_parameters_present_flag)?;
        if self.nalu.vui_parameters_present_flag {
            self.vui_parameters()?;
        }

        Ok(())
    }
}

impl<'n, W: Write> Synthesizer<'n, Pps, W> {
    pub fn synthesize(
        ref_idc: u8,
        pps: &'n Pps,
        writer: W,
        ep_enabled: bool,
    ) -> SynthesizerResult<()> {
        let mut s = Self { writer: NaluWriter::<W>::new(writer, ep_enabled), nalu: pps };

        s.writer.write_header(ref_idc, NaluType::Pps as u8)?;
        s.pic_parameter_set_rbsp()?;
        s.rbsp_trailing_bits()
    }

    fn pic_parameter_set_rbsp(&mut self) -> SynthesizerResult<()> {
        self.ue(self.nalu.pic_parameter_set_id)?;
        self.ue(self.nalu.seq_parameter_set_id)?;
        self.u(1, self.nalu.entropy_coding_mode_flag)?;
        self.u(1, self.nalu.bottom_field_pic_order_in_frame_present_flag)?;

        self.ue(self.nalu.num_slice_groups_minus1)?;
        if self.nalu.num_slice_groups_minus1 > 0 {
            return Err(SynthesizerError::Unsupported);
        }

        self.ue(self.nalu.num_ref_idx_l0_default_active_minus1)?;
        self.ue(self.nalu.num_ref_idx_l1_default_active_minus1)?;
        self.u(1, self.nalu.weighted_pred_flag)?;
        self.u(2, self.nalu.weighted_bipred_idc)?;
        self.se(self.nalu.pic_init_qp_minus26)?;
        self.se(self.nalu.pic_init_qs_minus26)?;
        self.se(self.nalu.chroma_qp_index_offset)?;
        self.u(1, self.nalu.deblocking_filter_control_present_flag)?;
        self.u(1, self.nalu.constrained_intra_pred_flag)?;
        self.u(1, self.nalu.redundant_pic_cnt_present_flag)?;

        if !(self.nalu.transform_8x8_mode_flag
            || self.nalu.pic_scaling_matrix_present_flag
            || self.nalu.second_chroma_qp_index_offset != 0)
        {
            return Ok(());
        }

        self.u(1, self.nalu.transform_8x8_mode_flag)?;
        self.u(1, self.nalu.pic_scaling_matrix_present_flag)?;

        if self.nalu.pic_scaling_matrix_present_flag {
            let mut scaling_list_count = 6;
            if self.nalu.transform_8x8_mode_flag {
                if self.nalu.sps.chroma_format_idc != 3 {
                    scaling_list_count += 2;
                } else {
                    scaling_list_count += 6;
                }
            }

            for i in 0..scaling_list_count {
                // Assume if scaling lists are zeroed that they are not present.
                if i < 6 {
                    if self.nalu.scaling_lists_4x4[i] == [0; 16] {
                        self.u(1, /* seq_scaling_list_present_flag */ false)?;
                    } else {
                        self.u(1, /* seq_scaling_list_present_flag */ true)?;
                        self.scaling_list(
                            &self.nalu.scaling_lists_4x4[i],
                            Self::default_scaling_list(i),
                        )?;
                    }
                } else if self.nalu.scaling_lists_8x8[i - 6] == [0; 64] {
                    self.u(1, /* seq_scaling_list_present_flag */ false)?;
                } else {
                    self.u(1, /* seq_scaling_list_present_flag */ true)?;
                    self.scaling_list(
                        &self.nalu.scaling_lists_8x8[i - 6],
                        Self::default_scaling_list(i),
                    )?;
                }
            }
        }

        self.se(self.nalu.second_chroma_qp_index_offset)?;

        Ok(())
    }
}

impl<'n, W: Write> Synthesizer<'n, SliceHeader, W> {
    /// Synthesizes a coded slice NAL unit up to its slice data: the 4-byte start code, the NAL
    /// unit header (`nal_ref_idc`, `nalu_type`) and `slice_header()` (H.264 7.3.3), with
    /// emulation prevention when `ep_enabled`. It is what an encoder hands the driver as a packed
    /// slice header (`VAEncPackedHeaderSlice`), which the driver continues with the slice data.
    ///
    /// `slice_header()` ends where the slice data begins, mid-byte, so the last byte is completed
    /// with zero bits; the function returns how many (0..=7). The packed header is
    /// `8 * bytes - padding` bits long.
    ///
    /// `nalu_type` is [`NaluType::SliceIdr`] or [`NaluType::Slice`]; `sps` and `pps` are the
    /// parameter sets the slice activates. `frame_num` and `pic_order_cnt_lsb` are written modulo
    /// `MaxFrameNum` and `MaxPicOrderCntLsb`, the numbers the stream counts with. Where a slice
    /// does not override them, the number of active references is the PPS's default.
    pub fn synthesize(
        nal_ref_idc: u8,
        nalu_type: NaluType,
        header: &'n SliceHeader,
        sps: &Sps,
        pps: &Pps,
        writer: W,
        ep_enabled: bool,
    ) -> SynthesizerResult<u8> {
        let idr = match nalu_type {
            NaluType::SliceIdr => true,
            NaluType::Slice => false,
            _ => return Err(SynthesizerError::Unsupported),
        };
        if idr && nal_ref_idc == 0 {
            return Err(SynthesizerError::Invalid("an IDR slice with nal_ref_idc 0 (7.4.1)"));
        }
        if nal_ref_idc > 3 {
            return Err(SynthesizerError::Invalid("nal_ref_idc past its 2 bits"));
        }
        // `slice_group_change_cycle` needs the slice group map, which the PPS synthesizer refuses
        // to write too.
        if pps.num_slice_groups_minus1 > 0 {
            return Err(SynthesizerError::Unsupported);
        }

        let mut s = Self { writer: NaluWriter::<W>::new(writer, ep_enabled), nalu: header };
        s.check_ranges(sps, pps)?;
        s.writer.write_header(nal_ref_idc, nalu_type as u8)?;
        s.slice_header(idr, nal_ref_idc, sps, pps)?;
        Ok(s.writer.flush_padded()?)
    }

    /// Refuses, before a bit is written, the values past the ranges the specification gives the
    /// elements this header is sized, indexed or shifted by. Past them the header would not read
    /// back as given (this crate's parser holds the same bounds), and `frame_num`'s and the POC
    /// lsb's moduli, the `pred_weight_table()` lists and its default weights would overflow.
    fn check_ranges(&self, sps: &Sps, pps: &Pps) -> SynthesizerResult<()> {
        let h = self.nalu;
        // 7.4.2.1.1: MaxFrameNum and MaxPicOrderCntLsb are at most 2^16.
        if sps.log2_max_frame_num_minus4 > 12 {
            return Err(SynthesizerError::Invalid("log2_max_frame_num_minus4 past 12 (7.4.2.1.1)"));
        }
        if sps.pic_order_cnt_type == 0 && sps.log2_max_pic_order_cnt_lsb_minus4 > 12 {
            return Err(SynthesizerError::Invalid(
                "log2_max_pic_order_cnt_lsb_minus4 past 12 (7.4.2.1.1)",
            ));
        }
        // 7.4.3: a sequence of frames alone has no `field_pic_flag`, which then reads as 0.
        if h.field_pic_flag && sps.frame_mbs_only_flag {
            return Err(SynthesizerError::Invalid(
                "a field slice in a frame-only sequence (7.4.3)",
            ));
        }
        if !Self::is_inter(h.slice_type) {
            return Ok(());
        }
        // 7.4.3: a list holds at most 16 references in a frame slice, 32 in a field slice. List 1
        // is a B slice's alone.
        let max_minus1 = if h.field_pic_flag { 31 } else { 15 };
        let (l0_minus1, l1_minus1) = self.num_ref_idx_active_minus1(pps);
        if l0_minus1 > max_minus1 || (h.slice_type.is_b() && l1_minus1 > max_minus1) {
            return Err(SynthesizerError::Invalid(
                "num_ref_idx_active_minus1 past 15 in a frame slice, 31 in a field slice (7.4.3)",
            ));
        }
        // 7.4.3.2: the chroma denominator is written, and so bounded, where chroma is coded.
        let pwt = &h.pred_weight_table;
        if self.has_pred_weight_table(pps)
            && (pwt.luma_log2_weight_denom > 7
                || (sps.chroma_array_type() != 0 && pwt.chroma_log2_weight_denom > 7))
        {
            return Err(SynthesizerError::Invalid("log2_weight_denom past 7 (7.4.3.2)"));
        }
        Ok(())
    }

    /// P, SP and B slices predict from list 0.
    fn is_inter(slice_type: SliceType) -> bool {
        !(slice_type.is_i() || slice_type.is_si())
    }

    /// Whether the slice carries `pred_weight_table()`: explicit weighted prediction (7.3.3).
    fn has_pred_weight_table(&self, pps: &Pps) -> bool {
        let slice_type = self.nalu.slice_type;
        (pps.weighted_pred_flag && (slice_type.is_p() || slice_type.is_sp()))
            || (pps.weighted_bipred_idc == 1 && slice_type.is_b())
    }

    /// The reference indices a list holds for this slice: the slice's own count when it
    /// overrides the PPS's, else the PPS's default.
    fn num_ref_idx_active_minus1(&self, pps: &Pps) -> (u8, u8) {
        let h = self.nalu;
        if h.num_ref_idx_active_override_flag {
            (h.num_ref_idx_l0_active_minus1, h.num_ref_idx_l1_active_minus1)
        } else {
            (pps.num_ref_idx_l0_default_active_minus1, pps.num_ref_idx_l1_default_active_minus1)
        }
    }

    /// H.264 7.3.3.
    fn slice_header(
        &mut self,
        idr: bool,
        nal_ref_idc: u8,
        sps: &Sps,
        pps: &Pps,
    ) -> SynthesizerResult<()> {
        let h = self.nalu;
        let slice_type = h.slice_type;
        let inter = Self::is_inter(slice_type);

        self.ue(h.first_mb_in_slice)?;
        self.ue(slice_type as u8)?;
        self.ue(h.pic_parameter_set_id)?;
        if sps.separate_colour_plane_flag {
            self.u(2, h.colour_plane_id)?;
        }

        let frame_num_bits = usize::from(sps.log2_max_frame_num_minus4) + 4;
        self.u(frame_num_bits, u32::from(h.frame_num) & (sps.max_frame_num() - 1))?;

        if !sps.frame_mbs_only_flag {
            self.u(1, h.field_pic_flag)?;
            if h.field_pic_flag {
                self.u(1, h.bottom_field_flag)?;
            }
        }

        if idr {
            self.ue(h.idr_pic_id)?;
        }

        let bottom_delta = pps.bottom_field_pic_order_in_frame_present_flag && !h.field_pic_flag;
        if sps.pic_order_cnt_type == 0 {
            let lsb_bits = usize::from(sps.log2_max_pic_order_cnt_lsb_minus4) + 4;
            self.u(lsb_bits, u32::from(h.pic_order_cnt_lsb) & ((1u32 << lsb_bits) - 1))?;
            if bottom_delta {
                self.se(h.delta_pic_order_cnt_bottom)?;
            }
        }
        if sps.pic_order_cnt_type == 1 && !sps.delta_pic_order_always_zero_flag {
            self.se(h.delta_pic_order_cnt[0])?;
            if bottom_delta {
                self.se(h.delta_pic_order_cnt[1])?;
            }
        }

        if pps.redundant_pic_cnt_present_flag {
            self.ue(h.redundant_pic_cnt)?;
        }

        if slice_type.is_b() {
            self.u(1, h.direct_spatial_mv_pred_flag)?;
        }

        if inter {
            self.u(1, h.num_ref_idx_active_override_flag)?;
            if h.num_ref_idx_active_override_flag {
                self.ue(h.num_ref_idx_l0_active_minus1)?;
                if slice_type.is_b() {
                    self.ue(h.num_ref_idx_l1_active_minus1)?;
                }
            }
        }

        // ref_pic_list_modification() (7.3.3.1); `nalu_type` is never 20 or 21 here, so
        // never the MVC form.
        if inter {
            self.ref_pic_list_modification(
                h.ref_pic_list_modification_flag_l0,
                &h.ref_pic_list_modification_l0,
            )?;
        }
        if slice_type.is_b() {
            self.ref_pic_list_modification(
                h.ref_pic_list_modification_flag_l1,
                &h.ref_pic_list_modification_l1,
            )?;
        }

        if self.has_pred_weight_table(pps) {
            self.pred_weight_table(sps, pps)?;
        }

        if nal_ref_idc != 0 {
            self.dec_ref_pic_marking(idr)?;
        }

        if pps.entropy_coding_mode_flag && inter {
            self.ue(h.cabac_init_idc)?;
        }

        self.se(h.slice_qp_delta)?;

        if slice_type.is_sp() || slice_type.is_si() {
            if slice_type.is_sp() {
                self.u(1, h.sp_for_switch_flag)?;
            }
            self.se(h.slice_qs_delta)?;
        }

        if pps.deblocking_filter_control_present_flag {
            self.ue(h.disable_deblocking_filter_idc)?;
            if h.disable_deblocking_filter_idc != 1 {
                self.se(h.slice_alpha_c0_offset_div2)?;
                self.se(h.slice_beta_offset_div2)?;
            }
        }

        Ok(())
    }

    /// One list's half of `ref_pic_list_modification()` (7.3.3.1). `ops` may or may not hold the
    /// closing `modification_of_pic_nums_idc` 3 (the parser keeps it); the list is closed once.
    fn ref_pic_list_modification(
        &mut self,
        flag: bool,
        ops: &[RefPicListModification],
    ) -> SynthesizerResult<()> {
        self.u(1, flag)?;
        if !flag {
            return Ok(());
        }
        for op in ops {
            match op.modification_of_pic_nums_idc {
                0 | 1 => {
                    self.ue(op.modification_of_pic_nums_idc)?;
                    self.ue(op.abs_diff_pic_num_minus1)?;
                }
                2 => {
                    self.ue(2u32)?;
                    self.ue(op.long_term_pic_num)?;
                }
                3 => break,
                // 4 and 5 are the MVC form, which a slice of nal_unit_type 1 or 5 cannot carry.
                _ => return Err(SynthesizerError::Invalid("modification_of_pic_nums_idc past 3")),
            }
        }
        self.ue(3u32)
    }

    /// `pred_weight_table()` (7.3.3.2). A weight flag is set where the entry differs from the
    /// value the flag's absence infers (weight `2^denom`, offset 0), so the table decodes to the
    /// values held. [`Self::check_ranges`] has bounded the list lengths and the denominators.
    fn pred_weight_table(&mut self, sps: &Sps, pps: &Pps) -> SynthesizerResult<()> {
        let h = self.nalu;
        let pwt = &h.pred_weight_table;
        let (l0_minus1, l1_minus1) = self.num_ref_idx_active_minus1(pps);

        self.ue(pwt.luma_log2_weight_denom)?;
        let luma_default = 1i32 << pwt.luma_log2_weight_denom;
        // The chroma weights, and their denominator, are coded where chroma is.
        let chroma_default =
            (sps.chroma_array_type() != 0).then(|| 1i32 << pwt.chroma_log2_weight_denom);
        if chroma_default.is_some() {
            self.ue(pwt.chroma_log2_weight_denom)?;
        }

        for i in 0..=usize::from(l0_minus1) {
            let (weight, offset) =
                (i32::from(pwt.luma_weight_l0[i]), i32::from(pwt.luma_offset_l0[i]));
            self.weight(weight, offset, luma_default)?;
            if let Some(chroma_default) = chroma_default {
                let w = pwt.chroma_weight_l0[i].map(i32::from);
                let o = pwt.chroma_offset_l0[i].map(i32::from);
                self.chroma_weights(w, o, chroma_default)?;
            }
        }
        if h.slice_type.is_b() {
            for i in 0..=usize::from(l1_minus1) {
                let (weight, offset) =
                    (i32::from(pwt.luma_weight_l1[i]), i32::from(pwt.luma_offset_l1[i]));
                self.weight(weight, offset, luma_default)?;
                if let Some(chroma_default) = chroma_default {
                    let w = pwt.chroma_weight_l1[i].map(i32::from);
                    let o = pwt.chroma_offset_l1[i].map(i32::from);
                    self.chroma_weights(w, o, chroma_default)?;
                }
            }
        }
        Ok(())
    }

    /// 7.4.3.2: a weight and an offset the table states are in -128..=127. (An inferred weight
    /// may be 128, `2^7`, which is why it is inferred.)
    fn check_explicit_weight(weight: i32, offset: i32) -> SynthesizerResult<()> {
        const RANGE: std::ops::RangeInclusive<i32> = -128..=127;
        if RANGE.contains(&weight) && RANGE.contains(&offset) {
            Ok(())
        } else {
            Err(SynthesizerError::Invalid("an explicit weight or offset past -128..=127 (7.4.3.2)"))
        }
    }

    /// One `luma_weight_lX_flag` and, when set, its weight and offset.
    fn weight(&mut self, weight: i32, offset: i32, default: i32) -> SynthesizerResult<()> {
        let explicit = weight != default || offset != 0;
        self.u(1, explicit)?;
        if explicit {
            Self::check_explicit_weight(weight, offset)?;
            self.se(weight)?;
            self.se(offset)?;
        }
        Ok(())
    }

    /// One `chroma_weight_lX_flag` and, when set, both components' weights and offsets.
    fn chroma_weights(
        &mut self,
        weight: [i32; 2],
        offset: [i32; 2],
        default: i32,
    ) -> SynthesizerResult<()> {
        let explicit = weight.iter().any(|w| *w != default) || offset.iter().any(|o| *o != 0);
        self.u(1, explicit)?;
        if explicit {
            for j in 0..2 {
                Self::check_explicit_weight(weight[j], offset[j])?;
                self.se(weight[j])?;
                self.se(offset[j])?;
            }
        }
        Ok(())
    }

    /// `dec_ref_pic_marking()` (7.3.3.3).
    fn dec_ref_pic_marking(&mut self, idr: bool) -> SynthesizerResult<()> {
        let marking = &self.nalu.dec_ref_pic_marking;
        if idr {
            self.u(1, marking.no_output_of_prior_pics_flag)?;
            self.u(1, marking.long_term_reference_flag)?;
            return Ok(());
        }
        self.u(1, marking.adaptive_ref_pic_marking_mode_flag)?;
        if !marking.adaptive_ref_pic_marking_mode_flag {
            return Ok(());
        }
        for op in &marking.inner {
            let mmco = op.memory_management_control_operation;
            match mmco {
                // 0 ends the list, which is closed once below.
                0 => break,
                1..=6 => {}
                _ => {
                    return Err(SynthesizerError::Invalid(
                        "memory_management_control_operation past 6",
                    ))
                }
            }
            self.ue(mmco)?;
            if mmco == 1 || mmco == 3 {
                self.ue(op.difference_of_pic_nums_minus1)?;
            }
            if mmco == 2 {
                self.ue(op.long_term_pic_num)?;
            }
            if mmco == 3 || mmco == 6 {
                self.ue(op.long_term_frame_idx)?;
            }
            if mmco == 4 {
                self.ue(op.max_long_term_frame_idx.to_value_plus1())?;
            }
        }
        self.ue(0u32)
    }
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::*;
    use crate::codec::h264::parser::Nalu;
    use crate::codec::h264::parser::NaluType;
    use crate::codec::h264::parser::Parser;
    use crate::codec::h264::parser::Profile;

    #[test]
    fn synthesize_sps() {
        let raw_sps_buf = [0x00, 0x00, 0x00, 0x01, 0x07, 0x00, 0x00, 0x0a, 0xfb, 0x88];
        let mut raw_sps = Cursor::new(&raw_sps_buf[..]);

        let nalu = Nalu::next(&mut raw_sps).unwrap();
        assert_eq!(nalu.header.type_, NaluType::Sps);

        let mut parser = Parser::default();
        let sps = parser.parse_sps(&nalu).unwrap();

        let mut buf = Vec::<u8>::new();
        Synthesizer::<'_, Sps, _>::synthesize(0, sps, &mut buf, false).unwrap();

        assert_eq!(buf, raw_sps_buf);

        let write_to_file = std::option_env!("CROS_CODECS_TEST_WRITE_TO_FILE") == Some("true");
        if write_to_file {
            let mut out = std::fs::File::create("sps.h264").unwrap();
            out.write_all(&buf).unwrap();
            out.flush().unwrap();
        }

        let mut cursor = Cursor::new(&buf[..]);
        let nalu = Nalu::next(&mut cursor).unwrap();

        let mut parser = Parser::default();

        let sps2 = parser.parse_sps(&nalu).unwrap();

        assert_eq!(sps, sps2);
    }

    #[test]
    fn synthesize_sps_scaling_lists() {
        let sps = Sps {
            profile_idc: Profile::High as u8,
            seq_scaling_matrix_present_flag: true,
            scaling_lists_4x4: [[11, 20, 10, 20, 10, 22, 10, 20, 10, 20, 13, 20, 10, 20, 10, 24];
                6],
            scaling_lists_8x8: [
                [
                    33, 20, 10, 21, 33, 20, 12, 20, 33, 23, 10, 20, 33, 20, 10, 20, 33, 24, 10, 20,
                    33, 20, 15, 20, 33, 20, 10, 26, 33, 20, 17, 20, 33, 28, 10, 20, 33, 20, 10, 20,
                    33, 29, 10, 20, 33, 20, 11, 20, 33, 20, 10, 20, 33, 20, 10, 20, 33, 20, 10, 20,
                    33, 20, 10, 20,
                ],
                [
                    10, 77, 11, 20, 10, 77, 12, 20, 10, 77, 13, 20, 10, 77, 14, 20, 10, 77, 15, 20,
                    10, 77, 16, 20, 10, 77, 17, 20, 10, 77, 18, 20, 10, 77, 19, 20, 10, 77, 10, 20,
                    10, 77, 10, 21, 10, 77, 10, 22, 10, 77, 10, 23, 10, 77, 10, 24, 10, 77, 10, 26,
                    10, 77, 10, 28,
                ],
                [0; 64],
                [0; 64],
                [0; 64],
                [0; 64],
            ],
            frame_mbs_only_flag: true,
            ..Default::default()
        };

        let mut buf = Vec::<u8>::new();
        Synthesizer::<'_, Sps, _>::synthesize(0, &sps, &mut buf, false).unwrap();

        let write_to_file = std::option_env!("CROS_CODECS_TEST_WRITE_TO_FILE") == Some("true");
        if write_to_file {
            let mut out = std::fs::File::create("sps.h264").unwrap();
            out.write_all(&buf).unwrap();
            out.flush().unwrap();
        }

        let mut cursor = Cursor::new(&buf[..]);
        let nalu = Nalu::next(&mut cursor).unwrap();

        let mut parser = Parser::default();

        let sps2 = parser.parse_sps(&nalu).unwrap();

        assert_eq!(sps.scaling_lists_4x4, sps2.scaling_lists_4x4);
        assert_eq!(sps.scaling_lists_8x8, sps2.scaling_lists_8x8);
    }

    #[test]
    fn synthesize_pps() {
        let raw_sps_pps = [
            0x00, 0x00, 0x00, 0x01, 0x07, 0x4d, 0x40, 0x0d, 0xa9, 0x18, 0x28, 0x3e, 0x60, 0x0d,
            0x41, 0x80, 0x41, 0xad, 0xb0, 0xad, 0x7b, 0xdf, 0x01, 0x00, 0x00, 0x00, 0x01, 0x08,
            0xde, 0x09, 0x88,
        ];

        let mut buf = Vec::<u8>::new();
        let mut out = Cursor::new(&mut buf);

        let mut cursor = Cursor::new(&raw_sps_pps[..]);
        let mut parser: Parser = Default::default();

        while let Ok(nalu) = Nalu::next(&mut cursor) {
            match nalu.header.type_ {
                NaluType::Sps => {
                    let sps = parser.parse_sps(&nalu).unwrap();
                    Synthesizer::<'_, Sps, _>::synthesize(0, sps, &mut out, false).unwrap();
                }
                NaluType::Pps => {
                    let pps = parser.parse_pps(&nalu).unwrap();
                    Synthesizer::<'_, Pps, _>::synthesize(0, pps, &mut out, false).unwrap();
                }
                _ => panic!(),
            }
        }

        let write_to_file = std::option_env!("CROS_CODECS_TEST_WRITE_TO_FILE") == Some("true");
        if write_to_file {
            let mut out = std::fs::File::create("sps_pps.h264").unwrap();
            out.write_all(&buf).unwrap();
            out.flush().unwrap();

            let mut out = std::fs::File::create("sps_pps_ref.h264").unwrap();
            out.write_all(&raw_sps_pps).unwrap();
            out.flush().unwrap();
        }

        assert_eq!(buf, raw_sps_pps);
    }
}
