// Copyright 2024 The ChromiumOS Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

//! An Annex B H.265 (HEVC) bitstream synthesizer.
//!
//! Emits spec-correct VPS/SPS/PPS RBSPs and slice-segment headers from the
//! [`crate::codec::h265::parser`] structs, for use as application-packed
//! headers (`VA_ENC_PACKED_HEADER_*`). It is the exact inverse of the parser:
//! every writer mirrors the corresponding parse function so the output
//! round-trips through the parser field-for-field.

use std::fmt;
use std::io::Write;

use crate::codec::h264::nalu_writer::NaluWriter;
use crate::codec::h264::nalu_writer::NaluWriterError;
use crate::codec::h265::parser::HrdParams;
use crate::codec::h265::parser::NaluType;
use crate::codec::h265::parser::Pps;
use crate::codec::h265::parser::PpsRangeExtension;
use crate::codec::h265::parser::PredWeightTable;
use crate::codec::h265::parser::ProfileTierLevel;
use crate::codec::h265::parser::RefPicListModification;
use crate::codec::h265::parser::ScalingLists;
use crate::codec::h265::parser::ShortTermRefPicSet;
use crate::codec::h265::parser::SliceHeader;
use crate::codec::h265::parser::SliceType;
use crate::codec::h265::parser::Sps;
use crate::codec::h265::parser::SpsRangeExtension;
use crate::codec::h265::parser::SublayerHrdParameters;
use crate::codec::h265::parser::VuiParams;
use crate::codec::h265::parser::Vps;

mod private {
    pub trait NaluStruct {}
}

impl private::NaluStruct for Vps {}
impl private::NaluStruct for Sps {}
impl private::NaluStruct for Pps {}
impl private::NaluStruct for SliceHeader {}

#[derive(Debug)]
pub enum SynthesizerError {
    Unsupported,
    NaluWriter(NaluWriterError),
}

impl fmt::Display for SynthesizerError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            SynthesizerError::Unsupported => write!(f, "tried to synthesize unsupported settings"),
            SynthesizerError::NaluWriter(x) => write!(f, "{}", x),
        }
    }
}

impl From<NaluWriterError> for SynthesizerError {
    fn from(err: NaluWriterError) -> Self {
        SynthesizerError::NaluWriter(err)
    }
}

pub type SynthesizerResult<T> = Result<T, SynthesizerError>;

/// Extended Sample Aspect Ratio - H.265 Table E-1.
const EXTENDED_SAR: u32 = 255;

/// A helper to output typed HEVC NALUs to [`std::io::Write`] using [`NaluWriter`].
pub struct Synthesizer<'n, N: private::NaluStruct, W: Write> {
    writer: NaluWriter<W>,
    nalu: &'n N,
}

/// Returns `true` if `general_profile_idc`/`sub_layer_profile_idc` equals any of
/// `candidates`, or if the corresponding `..._profile_compatibility_flag` is
/// set. Mirrors the branch conditions in `Parser::parse_profile_tier_level`.
fn profile_match(idc: u8, compat: &[bool; 32], candidates: &[usize]) -> bool {
    candidates.iter().any(|&v| idc as usize == v || compat[v])
}

/// `Ceil(Log2(n))`, matching the parser's `(n as f64).log2().ceil()` for
/// variable-length field widths.
fn ceil_log2(n: u32) -> usize {
    (n as f64).log2().ceil() as usize
}

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

    /// Writes `bits` reserved zero bits.
    fn reserved_zero(&mut self, mut bits: usize) -> SynthesizerResult<()> {
        while bits > 0 {
            let n = std::cmp::min(bits, 32);
            self.f(n, 0u32)?;
            bits -= n;
        }
        Ok(())
    }

    fn rbsp_trailing_bits(&mut self) -> SynthesizerResult<()> {
        self.f(1, 1u32)?;

        while !self.writer.aligned() {
            self.f(1, 0u32)?;
        }

        Ok(())
    }

    /// `profile_tier_level()` — H.265 7.3.3.
    fn profile_tier_level(
        &mut self,
        ptl: &ProfileTierLevel,
        profile_present_flag: bool,
        max_sub_layers_minus1: u8,
    ) -> SynthesizerResult<()> {
        if profile_present_flag {
            self.u(2, ptl.general_profile_space)?;
            self.u(1, ptl.general_tier_flag)?;
            self.u(5, ptl.general_profile_idc)?;

            for i in 0..32 {
                self.u(1, ptl.general_profile_compatibility_flag[i])?;
            }

            self.u(1, ptl.general_progressive_source_flag)?;
            self.u(1, ptl.general_interlaced_source_flag)?;
            self.u(1, ptl.general_non_packed_constraint_flag)?;
            self.u(1, ptl.general_frame_only_constraint_flag)?;

            let idc = ptl.general_profile_idc;
            let compat = &ptl.general_profile_compatibility_flag;

            // The constraint-flags region is always exactly 43 bits.
            if profile_match(idc, compat, &[4, 5, 6, 7, 8, 9, 10, 11]) {
                self.u(1, ptl.general_max_12bit_constraint_flag)?;
                self.u(1, ptl.general_max_10bit_constraint_flag)?;
                self.u(1, ptl.general_max_8bit_constraint_flag)?;
                self.u(1, ptl.general_max_422chroma_constraint_flag)?;
                self.u(1, ptl.general_max_420chroma_constraint_flag)?;
                self.u(1, ptl.general_max_monochrome_constraint_flag)?;
                self.u(1, ptl.general_intra_constraint_flag)?;
                self.u(1, ptl.general_one_picture_only_constraint_flag)?;
                self.u(1, ptl.general_lower_bit_rate_constraint_flag)?;
                if profile_match(idc, compat, &[5, 9, 10, 11]) {
                    self.u(1, ptl.general_max_14bit_constraint_flag)?;
                    self.reserved_zero(33)?; // general_reserved_zero_33bits
                } else {
                    self.reserved_zero(34)?; // general_reserved_zero_34bits
                }
            } else if profile_match(idc, compat, &[2]) {
                self.reserved_zero(7)?; // general_reserved_zero_7bits
                self.u(1, ptl.general_one_picture_only_constraint_flag)?;
                self.reserved_zero(35)?; // general_reserved_zero_35bits
            } else {
                self.reserved_zero(43)?; // general_reserved_zero_43bits
            }

            // general_inbld_flag / general_reserved_zero_bit (always 1 bit).
            if profile_match(idc, compat, &[1, 2, 3, 4, 5, 9, 11]) {
                self.u(1, ptl.general_inbld_flag)?;
            } else {
                self.reserved_zero(1)?;
            }
        }

        self.u(8, ptl.general_level_idc as u8)?;

        for i in 0..max_sub_layers_minus1 as usize {
            self.u(1, ptl.sub_layer_profile_present_flag[i])?;
            self.u(1, ptl.sub_layer_level_present_flag[i])?;
        }

        if max_sub_layers_minus1 > 0 {
            for _ in max_sub_layers_minus1..8 {
                self.reserved_zero(2)?; // reserved_zero_2bits
            }
        }

        for i in 0..max_sub_layers_minus1 as usize {
            // §7.3.3: the sub-layer profile block gates on
            // sub_layer_profile_present_flag[i]; the level field below gates on
            // sub_layer_level_present_flag[i].
            if ptl.sub_layer_profile_present_flag[i] {
                self.u(2, ptl.sub_layer_profile_space[i])?;
                self.u(1, ptl.sub_layer_tier_flag[i])?;
                self.u(5, ptl.sub_layer_profile_idc[i])?;
                for j in 0..32 {
                    self.u(1, ptl.sub_layer_profile_compatibility_flag[i][j])?;
                }
                self.u(1, ptl.sub_layer_progressive_source_flag[i])?;
                self.u(1, ptl.sub_layer_interlaced_source_flag[i])?;
                self.u(1, ptl.sub_layer_non_packed_constraint_flag[i])?;
                self.u(1, ptl.sub_layer_frame_only_constraint_flag[i])?;

                let idc = ptl.sub_layer_profile_idc[i];
                let compat = &ptl.sub_layer_profile_compatibility_flag[i];

                if profile_match(idc, compat, &[4, 5, 6, 7, 8, 9, 10, 11]) {
                    self.u(1, ptl.sub_layer_max_12bit_constraint_flag[i])?;
                    self.u(1, ptl.sub_layer_max_10bit_constraint_flag[i])?;
                    self.u(1, ptl.sub_layer_max_8bit_constraint_flag[i])?;
                    self.u(1, ptl.sub_layer_max_422chroma_constraint_flag[i])?;
                    self.u(1, ptl.sub_layer_max_420chroma_constraint_flag[i])?;
                    self.u(1, ptl.sub_layer_max_monochrome_constraint_flag[i])?;
                    self.u(1, ptl.sub_layer_intra_constraint_flag[i])?;
                    self.u(1, ptl.sub_layer_one_picture_only_constraint_flag[i])?;
                    self.u(1, ptl.sub_layer_lower_bit_rate_constraint_flag[i])?;
                    if profile_match(idc, compat, &[5, 9, 10, 11]) {
                        self.u(1, ptl.sub_layer_max_14bit_constraint_flag[i])?;
                        self.reserved_zero(33)?;
                    } else {
                        self.reserved_zero(34)?;
                    }
                } else if profile_match(idc, compat, &[2]) {
                    self.reserved_zero(7)?;
                    self.u(1, ptl.sub_layer_one_picture_only_constraint_flag[i])?;
                    self.reserved_zero(35)?;
                } else {
                    self.reserved_zero(43)?;
                }

                if profile_match(idc, compat, &[1, 2, 3, 4, 5, 9, 11]) {
                    self.u(1, ptl.sub_layer_inbld_flag[i])?;
                } else {
                    self.reserved_zero(1)?;
                }
            }

            if ptl.sub_layer_level_present_flag[i] {
                self.u(8, ptl.sub_layer_level_idc[i] as u8)?;
            }
        }

        Ok(())
    }

    /// `scaling_list_data()` — H.265 7.3.4. Always emitted in explicit form
    /// (`scaling_list_pred_mode_flag == 1`), which round-trips any coefficient
    /// set through the parser.
    fn scaling_list_data(&mut self, sl: &ScalingLists) -> SynthesizerResult<()> {
        for size_id in 0..4usize {
            let mut matrix_id = 0usize;
            while matrix_id < 6 {
                // scaling_list_pred_mode_flag = 1 (explicit coefficients).
                self.u(1, true)?;

                let coef_num = std::cmp::min(64usize, 1 << (4 + (size_id << 1)));
                let mut next_coef = 8i32;

                if size_id > 1 {
                    let dc = if size_id == 2 {
                        sl.scaling_list_dc_coef_minus8_16x16[matrix_id]
                    } else {
                        sl.scaling_list_dc_coef_minus8_32x32[matrix_id]
                    };
                    self.se(dc as i32)?;
                    next_coef = dc as i32 + 8;
                }

                for i in 0..coef_num {
                    let coef = match size_id {
                        0 => sl.scaling_list_4x4[matrix_id][i],
                        1 => sl.scaling_list_8x8[matrix_id][i],
                        2 => sl.scaling_list_16x16[matrix_id][i],
                        _ => sl.scaling_list_32x32[matrix_id][i],
                    } as i32;

                    // Invert `(next_coef + delta + 256) % 256 == coef`, choosing
                    // the representative `delta` in [-128, 127].
                    let delta = (coef - next_coef).rem_euclid(256);
                    let delta = if delta >= 128 { delta - 256 } else { delta };
                    self.se(delta)?;
                    next_coef = coef;
                }

                matrix_id += if size_id == 3 { 3 } else { 1 };
            }
        }

        Ok(())
    }

    /// `st_ref_pic_set( stRpsIdx )` — H.265 7.3.7. Always emitted in explicit
    /// form (`inter_ref_pic_set_prediction_flag == 0`); the LowDelay-IPPP
    /// encoder never uses inter-RPS prediction.
    fn short_term_ref_pic_set(
        &mut self,
        st: &ShortTermRefPicSet,
        st_rps_idx: u8,
    ) -> SynthesizerResult<()> {
        if st_rps_idx != 0 {
            self.u(1, st.inter_ref_pic_set_prediction_flag)?;
        }

        if st.inter_ref_pic_set_prediction_flag {
            // Only the explicit form is supported (and produced by the encoder).
            return Err(SynthesizerError::Unsupported);
        }

        self.ue(u32::from(st.num_negative_pics))?;
        self.ue(u32::from(st.num_positive_pics))?;

        for i in 0..st.num_negative_pics as usize {
            let delta_poc_s0_minus1 = if i == 0 {
                -st.delta_poc_s0[0] - 1
            } else {
                st.delta_poc_s0[i - 1] - st.delta_poc_s0[i] - 1
            };
            self.ue(delta_poc_s0_minus1 as u32)?;
            self.u(1, st.used_by_curr_pic_s0[i])?;
        }

        for i in 0..st.num_positive_pics as usize {
            let delta_poc_s1_minus1 = if i == 0 {
                st.delta_poc_s1[0] - 1
            } else {
                st.delta_poc_s1[i] - st.delta_poc_s1[i - 1] - 1
            };
            self.ue(delta_poc_s1_minus1 as u32)?;
            self.u(1, st.used_by_curr_pic_s1[i])?;
        }

        Ok(())
    }

    /// `sub_layer_hrd_parameters( subLayerId )` — H.265 E.2.3.
    fn sublayer_hrd_parameters(
        &mut self,
        h: &SublayerHrdParameters,
        cpb_cnt: u32,
        sub_pic_hrd_params_present_flag: bool,
    ) -> SynthesizerResult<()> {
        for i in 0..cpb_cnt as usize {
            self.ue(h.bit_rate_value_minus1[i])?;
            self.ue(h.cpb_size_value_minus1[i])?;
            if sub_pic_hrd_params_present_flag {
                self.ue(h.cpb_size_du_value_minus1[i])?;
                self.ue(h.bit_rate_du_value_minus1[i])?;
            }
            self.u(1, h.cbr_flag[i])?;
        }

        Ok(())
    }

    /// `hrd_parameters( commonInfPresentFlag, maxNumSubLayersMinus1 )` —
    /// H.265 E.2.2.
    fn hrd_parameters(
        &mut self,
        hrd: &HrdParams,
        common_inf_present_flag: bool,
        max_num_sublayers_minus1: u8,
    ) -> SynthesizerResult<()> {
        if common_inf_present_flag {
            self.u(1, hrd.nal_hrd_parameters_present_flag)?;
            self.u(1, hrd.vcl_hrd_parameters_present_flag)?;
            if hrd.nal_hrd_parameters_present_flag || hrd.vcl_hrd_parameters_present_flag {
                self.u(1, hrd.sub_pic_hrd_params_present_flag)?;
                if hrd.sub_pic_hrd_params_present_flag {
                    self.u(8, hrd.tick_divisor_minus2)?;
                    self.u(5, hrd.du_cpb_removal_delay_increment_length_minus1)?;
                    self.u(1, hrd.sub_pic_cpb_params_in_pic_timing_sei_flag)?;
                    self.u(5, hrd.dpb_output_delay_du_length_minus1)?;
                }
                self.u(4, hrd.bit_rate_scale)?;
                self.u(4, hrd.cpb_size_scale)?;
                if hrd.sub_pic_hrd_params_present_flag {
                    self.u(4, hrd.cpb_size_du_scale)?;
                }
                self.u(5, hrd.initial_cpb_removal_delay_length_minus1)?;
                self.u(5, hrd.au_cpb_removal_delay_length_minus1)?;
                self.u(5, hrd.dpb_output_delay_length_minus1)?;
            }
        }

        for i in 0..=max_num_sublayers_minus1 as usize {
            self.u(1, hrd.fixed_pic_rate_general_flag[i])?;
            // §E.2.2: fixed_pic_rate_within_cvs_flag[i] is only coded when
            // fixed_pic_rate_general_flag[i] == 0; when general == 1 it is
            // inferred to be 1, and the branch below uses the inferred value.
            let within_cvs = if !hrd.fixed_pic_rate_general_flag[i] {
                self.u(1, hrd.fixed_pic_rate_within_cvs_flag[i])?;
                hrd.fixed_pic_rate_within_cvs_flag[i]
            } else {
                true
            };
            if within_cvs {
                self.ue(hrd.elemental_duration_in_tc_minus1[i])?;
            } else {
                self.u(1, hrd.low_delay_hrd_flag[i])?;
            }
            if !hrd.low_delay_hrd_flag[i] {
                self.ue(hrd.cpb_cnt_minus1[i])?;
            }
            if hrd.nal_hrd_parameters_present_flag {
                self.sublayer_hrd_parameters(
                    &hrd.nal_hrd[i],
                    hrd.cpb_cnt_minus1[i] + 1,
                    hrd.sub_pic_hrd_params_present_flag,
                )?;
            }
            if hrd.vcl_hrd_parameters_present_flag {
                self.sublayer_hrd_parameters(
                    &hrd.vcl_hrd[i],
                    hrd.cpb_cnt_minus1[i] + 1,
                    hrd.sub_pic_hrd_params_present_flag,
                )?;
            }
        }

        Ok(())
    }

    /// `vui_parameters()` — H.265 E.2.1.
    fn vui_parameters(
        &mut self,
        vui: &VuiParams,
        sps_max_sub_layers_minus1: u8,
    ) -> SynthesizerResult<()> {
        self.u(1, vui.aspect_ratio_info_present_flag)?;
        if vui.aspect_ratio_info_present_flag {
            self.u(8, vui.aspect_ratio_idc)?;
            if vui.aspect_ratio_idc == EXTENDED_SAR {
                self.u(16, vui.sar_width)?;
                self.u(16, vui.sar_height)?;
            }
        }

        self.u(1, vui.overscan_info_present_flag)?;
        if vui.overscan_info_present_flag {
            self.u(1, vui.overscan_appropriate_flag)?;
        }

        self.u(1, vui.video_signal_type_present_flag)?;
        if vui.video_signal_type_present_flag {
            self.u(3, vui.video_format)?;
            self.u(1, vui.video_full_range_flag)?;
            self.u(1, vui.colour_description_present_flag)?;
            if vui.colour_description_present_flag {
                self.u(8, vui.colour_primaries)?;
                self.u(8, vui.transfer_characteristics)?;
                self.u(8, vui.matrix_coeffs)?;
            }
        }

        self.u(1, vui.chroma_loc_info_present_flag)?;
        if vui.chroma_loc_info_present_flag {
            self.ue(vui.chroma_sample_loc_type_top_field)?;
            self.ue(vui.chroma_sample_loc_type_bottom_field)?;
        }

        self.u(1, vui.neutral_chroma_indication_flag)?;
        self.u(1, vui.field_seq_flag)?;
        self.u(1, vui.frame_field_info_present_flag)?;
        self.u(1, vui.default_display_window_flag)?;
        if vui.default_display_window_flag {
            self.ue(vui.def_disp_win_left_offset)?;
            self.ue(vui.def_disp_win_right_offset)?;
            self.ue(vui.def_disp_win_top_offset)?;
            self.ue(vui.def_disp_win_bottom_offset)?;
        }

        self.u(1, vui.timing_info_present_flag)?;
        if vui.timing_info_present_flag {
            self.u(32, vui.num_units_in_tick)?;
            self.u(32, vui.time_scale)?;
            self.u(1, vui.poc_proportional_to_timing_flag)?;
            if vui.poc_proportional_to_timing_flag {
                self.ue(vui.num_ticks_poc_diff_one_minus1)?;
            }
            self.u(1, vui.hrd_parameters_present_flag)?;
            if vui.hrd_parameters_present_flag {
                self.hrd_parameters(&vui.hrd, true, sps_max_sub_layers_minus1)?;
            }
        }

        self.u(1, vui.bitstream_restriction_flag)?;
        if vui.bitstream_restriction_flag {
            self.u(1, vui.tiles_fixed_structure_flag)?;
            self.u(1, vui.motion_vectors_over_pic_boundaries_flag)?;
            self.u(1, vui.restricted_ref_pic_lists_flag)?;
            self.ue(vui.min_spatial_segmentation_idc)?;
            self.ue(vui.max_bytes_per_pic_denom)?;
            self.ue(vui.max_bits_per_min_cu_denom)?;
            self.ue(vui.log2_max_mv_length_horizontal)?;
            self.ue(vui.log2_max_mv_length_vertical)?;
        }

        Ok(())
    }

    /// `sps_range_extension()` — H.265 7.3.2.2.2.
    fn sps_range_extension(&mut self, ext: &SpsRangeExtension) -> SynthesizerResult<()> {
        self.u(1, ext.transform_skip_rotation_enabled_flag)?;
        self.u(1, ext.transform_skip_context_enabled_flag)?;
        self.u(1, ext.implicit_rdpcm_enabled_flag)?;
        self.u(1, ext.explicit_rdpcm_enabled_flag)?;
        self.u(1, ext.extended_precision_processing_flag)?;
        self.u(1, ext.intra_smoothing_disabled_flag)?;
        self.u(1, ext.high_precision_offsets_enabled_flag)?;
        self.u(1, ext.persistent_rice_adaptation_enabled_flag)?;
        self.u(1, ext.cabac_bypass_alignment_enabled_flag)?;
        Ok(())
    }
}

impl<'n, W: Write> Synthesizer<'n, Vps, W> {
    /// Synthesizes a complete VPS NALU (start code + 2-byte NAL header + RBSP)
    /// into `writer`.
    pub fn synthesize(vps: &'n Vps, writer: W, ep_enabled: bool) -> SynthesizerResult<()> {
        let mut s = Self { writer: NaluWriter::<W>::new(writer, ep_enabled), nalu: vps };

        s.writer.write_header_hevc(NaluType::VpsNut as u8, 0, 1)?;
        s.video_parameter_set_rbsp()?;
        s.rbsp_trailing_bits()
    }

    /// `video_parameter_set_rbsp()` — H.265 7.3.2.1.
    fn video_parameter_set_rbsp(&mut self) -> SynthesizerResult<()> {
        let vps = self.nalu;

        self.u(4, vps.video_parameter_set_id)?;
        self.u(1, vps.base_layer_internal_flag)?;
        self.u(1, vps.base_layer_available_flag)?;
        self.u(6, vps.max_layers_minus1)?;
        self.u(3, vps.max_sub_layers_minus1)?;
        self.u(1, vps.temporal_id_nesting_flag)?;
        self.u(16, /* vps_reserved_0xffff_16bits */ 0xffffu32)?;

        self.profile_tier_level(&vps.profile_tier_level, true, vps.max_sub_layers_minus1)?;

        self.u(1, vps.sub_layer_ordering_info_present_flag)?;
        let start = if vps.sub_layer_ordering_info_present_flag {
            0
        } else {
            vps.max_sub_layers_minus1 as usize
        };
        for i in start..=vps.max_sub_layers_minus1 as usize {
            self.ue(vps.max_dec_pic_buffering_minus1[i])?;
            self.ue(vps.max_num_reorder_pics[i])?;
            self.ue(vps.max_latency_increase_plus1[i])?;
        }

        self.u(6, vps.max_layer_id)?;
        self.ue(vps.num_layer_sets_minus1)?;
        for _ in 1..=vps.num_layer_sets_minus1 {
            for _ in 0..=vps.max_layer_id {
                // layer_id_included_flag[i][j] — always 0 for a single layer set.
                self.u(1, false)?;
            }
        }

        self.u(1, vps.timing_info_present_flag)?;
        if vps.timing_info_present_flag {
            self.u(32, vps.num_units_in_tick)?;
            self.u(32, vps.time_scale)?;
            self.u(1, vps.poc_proportional_to_timing_flag)?;
            if vps.poc_proportional_to_timing_flag {
                self.ue(vps.num_ticks_poc_diff_one_minus1)?;
            }
            self.ue(vps.num_hrd_parameters)?;
            for i in 0..vps.num_hrd_parameters as usize {
                self.ue(vps.hrd_layer_set_idx[i])?;
                let cprms_present = if i > 0 {
                    self.u(1, vps.cprms_present_flag[i])?;
                    vps.cprms_present_flag[i]
                } else {
                    true
                };
                self.hrd_parameters(&vps.hrd_parameters[i], cprms_present, vps.max_sub_layers_minus1)?;
            }
        }

        self.u(1, vps.extension_flag)?;

        Ok(())
    }
}

impl<'n, W: Write> Synthesizer<'n, Sps, W> {
    /// Synthesizes a complete SPS NALU (start code + 2-byte NAL header + RBSP)
    /// into `writer`.
    pub fn synthesize(sps: &'n Sps, writer: W, ep_enabled: bool) -> SynthesizerResult<()> {
        let mut s = Self { writer: NaluWriter::<W>::new(writer, ep_enabled), nalu: sps };

        s.writer.write_header_hevc(NaluType::SpsNut as u8, 0, 1)?;
        s.seq_parameter_set_rbsp()?;
        s.rbsp_trailing_bits()
    }

    /// `seq_parameter_set_rbsp()` — H.265 7.3.2.2.
    fn seq_parameter_set_rbsp(&mut self) -> SynthesizerResult<()> {
        let sps = self.nalu;

        self.u(4, sps.video_parameter_set_id)?;
        self.u(3, sps.max_sub_layers_minus1)?;
        self.u(1, sps.temporal_id_nesting_flag)?;

        self.profile_tier_level(&sps.profile_tier_level, true, sps.max_sub_layers_minus1)?;

        self.ue(sps.seq_parameter_set_id)?;
        self.ue(sps.chroma_format_idc)?;
        if sps.chroma_format_idc == 3 {
            self.u(1, sps.separate_colour_plane_flag)?;
        }

        self.ue(sps.pic_width_in_luma_samples)?;
        self.ue(sps.pic_height_in_luma_samples)?;

        self.u(1, sps.conformance_window_flag)?;
        if sps.conformance_window_flag {
            self.ue(sps.conf_win_left_offset)?;
            self.ue(sps.conf_win_right_offset)?;
            self.ue(sps.conf_win_top_offset)?;
            self.ue(sps.conf_win_bottom_offset)?;
        }

        self.ue(sps.bit_depth_luma_minus8)?;
        self.ue(sps.bit_depth_chroma_minus8)?;
        self.ue(sps.log2_max_pic_order_cnt_lsb_minus4)?;

        self.u(1, sps.sub_layer_ordering_info_present_flag)?;
        let start = if sps.sub_layer_ordering_info_present_flag {
            0
        } else {
            sps.max_sub_layers_minus1 as usize
        };
        for i in start..=sps.max_sub_layers_minus1 as usize {
            self.ue(sps.max_dec_pic_buffering_minus1[i])?;
            self.ue(sps.max_num_reorder_pics[i])?;
            self.ue(sps.max_latency_increase_plus1[i])?;
        }

        self.ue(sps.log2_min_luma_coding_block_size_minus3)?;
        self.ue(sps.log2_diff_max_min_luma_coding_block_size)?;
        self.ue(sps.log2_min_luma_transform_block_size_minus2)?;
        self.ue(sps.log2_diff_max_min_luma_transform_block_size)?;
        self.ue(sps.max_transform_hierarchy_depth_inter)?;
        self.ue(sps.max_transform_hierarchy_depth_intra)?;

        self.u(1, sps.scaling_list_enabled_flag)?;
        if sps.scaling_list_enabled_flag {
            self.u(1, sps.scaling_list_data_present_flag)?;
            if sps.scaling_list_data_present_flag {
                self.scaling_list_data(&sps.scaling_list)?;
            }
        }

        self.u(1, sps.amp_enabled_flag)?;
        self.u(1, sps.sample_adaptive_offset_enabled_flag)?;

        self.u(1, sps.pcm_enabled_flag)?;
        if sps.pcm_enabled_flag {
            self.u(4, sps.pcm_sample_bit_depth_luma_minus1)?;
            self.u(4, sps.pcm_sample_bit_depth_chroma_minus1)?;
            self.ue(sps.log2_min_pcm_luma_coding_block_size_minus3)?;
            self.ue(sps.log2_diff_max_min_pcm_luma_coding_block_size)?;
            self.u(1, sps.pcm_loop_filter_disabled_flag)?;
        }

        self.ue(sps.num_short_term_ref_pic_sets)?;
        for i in 0..sps.num_short_term_ref_pic_sets {
            self.short_term_ref_pic_set(&sps.short_term_ref_pic_set[i as usize], i)?;
        }

        self.u(1, sps.long_term_ref_pics_present_flag)?;
        if sps.long_term_ref_pics_present_flag {
            self.ue(sps.num_long_term_ref_pics_sps)?;
            for i in 0..sps.num_long_term_ref_pics_sps as usize {
                self.u(
                    usize::from(sps.log2_max_pic_order_cnt_lsb_minus4) + 4,
                    sps.lt_ref_pic_poc_lsb_sps[i],
                )?;
                self.u(1, sps.used_by_curr_pic_lt_sps_flag[i])?;
            }
        }

        self.u(1, sps.temporal_mvp_enabled_flag)?;
        self.u(1, sps.strong_intra_smoothing_enabled_flag)?;

        self.u(1, sps.vui_parameters_present_flag)?;
        if sps.vui_parameters_present_flag {
            self.vui_parameters(&sps.vui_parameters, sps.max_sub_layers_minus1)?;
        }

        self.u(1, sps.extension_present_flag)?;
        if sps.extension_present_flag {
            self.u(1, sps.range_extension_flag)?;
            if sps.range_extension_flag {
                self.sps_range_extension(&sps.range_extension)?;
            }
            // sps_multilayer_extension_flag / sps_3d_extension_flag: unsupported.
            self.u(1, false)?;
            self.u(1, false)?;
            self.u(1, sps.scc_extension_flag)?;
            if sps.scc_extension_flag {
                return Err(SynthesizerError::Unsupported);
            }
            self.u(4, /* sps_extension_4bits */ 0u32)?;
        }

        Ok(())
    }
}

impl<'n, W: Write> Synthesizer<'n, Pps, W> {
    /// Synthesizes a complete PPS NALU (start code + 2-byte NAL header + RBSP)
    /// into `writer`.
    pub fn synthesize(pps: &'n Pps, writer: W, ep_enabled: bool) -> SynthesizerResult<()> {
        let mut s = Self { writer: NaluWriter::<W>::new(writer, ep_enabled), nalu: pps };

        s.writer.write_header_hevc(NaluType::PpsNut as u8, 0, 1)?;
        s.pic_parameter_set_rbsp()?;
        s.rbsp_trailing_bits()
    }

    /// `pps_range_extension()` — H.265 7.3.2.3.2.
    fn pps_range_extension(
        &mut self,
        ext: &PpsRangeExtension,
        transform_skip_enabled_flag: bool,
    ) -> SynthesizerResult<()> {
        if transform_skip_enabled_flag {
            self.ue(ext.log2_max_transform_skip_block_size_minus2)?;
        }
        self.u(1, ext.cross_component_prediction_enabled_flag)?;
        self.u(1, ext.chroma_qp_offset_list_enabled_flag)?;
        if ext.chroma_qp_offset_list_enabled_flag {
            self.ue(ext.diff_cu_chroma_qp_offset_depth)?;
            self.ue(ext.chroma_qp_offset_list_len_minus1)?;
            for i in 0..=ext.chroma_qp_offset_list_len_minus1 as usize {
                self.se(ext.cb_qp_offset_list[i])?;
                self.se(ext.cr_qp_offset_list[i])?;
            }
        }
        self.ue(ext.log2_sao_offset_scale_luma)?;
        self.ue(ext.log2_sao_offset_scale_chroma)?;
        Ok(())
    }

    /// `pic_parameter_set_rbsp()` — H.265 7.3.2.3.1.
    fn pic_parameter_set_rbsp(&mut self) -> SynthesizerResult<()> {
        let pps = self.nalu;

        self.ue(pps.pic_parameter_set_id)?;
        self.ue(pps.seq_parameter_set_id)?;
        self.u(1, pps.dependent_slice_segments_enabled_flag)?;
        self.u(1, pps.output_flag_present_flag)?;
        self.u(3, pps.num_extra_slice_header_bits)?;
        self.u(1, pps.sign_data_hiding_enabled_flag)?;
        self.u(1, pps.cabac_init_present_flag)?;
        self.ue(pps.num_ref_idx_l0_default_active_minus1)?;
        self.ue(pps.num_ref_idx_l1_default_active_minus1)?;
        self.se(pps.init_qp_minus26)?;
        self.u(1, pps.constrained_intra_pred_flag)?;
        self.u(1, pps.transform_skip_enabled_flag)?;
        self.u(1, pps.cu_qp_delta_enabled_flag)?;
        if pps.cu_qp_delta_enabled_flag {
            self.ue(pps.diff_cu_qp_delta_depth)?;
        }
        self.se(pps.cb_qp_offset)?;
        self.se(pps.cr_qp_offset)?;
        self.u(1, pps.slice_chroma_qp_offsets_present_flag)?;
        self.u(1, pps.weighted_pred_flag)?;
        self.u(1, pps.weighted_bipred_flag)?;
        self.u(1, pps.transquant_bypass_enabled_flag)?;
        self.u(1, pps.tiles_enabled_flag)?;
        self.u(1, pps.entropy_coding_sync_enabled_flag)?;

        if pps.tiles_enabled_flag {
            self.ue(pps.num_tile_columns_minus1)?;
            self.ue(pps.num_tile_rows_minus1)?;
            self.u(1, pps.uniform_spacing_flag)?;
            if !pps.uniform_spacing_flag {
                for i in 0..pps.num_tile_columns_minus1 as usize {
                    self.ue(pps.column_width_minus1[i])?;
                }
                for i in 0..pps.num_tile_rows_minus1 as usize {
                    self.ue(pps.row_height_minus1[i])?;
                }
            }
            self.u(1, pps.loop_filter_across_tiles_enabled_flag)?;
        }

        self.u(1, pps.loop_filter_across_slices_enabled_flag)?;
        self.u(1, pps.deblocking_filter_control_present_flag)?;
        if pps.deblocking_filter_control_present_flag {
            self.u(1, pps.deblocking_filter_override_enabled_flag)?;
            self.u(1, pps.deblocking_filter_disabled_flag)?;
            if !pps.deblocking_filter_disabled_flag {
                self.se(pps.beta_offset_div2)?;
                self.se(pps.tc_offset_div2)?;
            }
        }

        self.u(1, pps.scaling_list_data_present_flag)?;
        if pps.scaling_list_data_present_flag {
            self.scaling_list_data(&pps.scaling_list)?;
        }

        self.u(1, pps.lists_modification_present_flag)?;
        self.ue(pps.log2_parallel_merge_level_minus2)?;
        self.u(1, pps.slice_segment_header_extension_present_flag)?;

        self.u(1, pps.extension_present_flag)?;
        if pps.extension_present_flag {
            self.u(1, pps.range_extension_flag)?;
            if pps.range_extension_flag {
                self.pps_range_extension(&pps.range_extension, pps.transform_skip_enabled_flag)?;
            }
            // pps_multilayer_extension_flag / pps_3d_extension_flag: unsupported.
            self.u(1, false)?;
            self.u(1, false)?;
            self.u(1, pps.scc_extension_flag)?;
            if pps.scc_extension_flag {
                return Err(SynthesizerError::Unsupported);
            }
            self.u(4, /* pps_extension_4bits */ 0u32)?;
        }

        Ok(())
    }
}

impl<'n, W: Write> Synthesizer<'n, SliceHeader, W> {
    /// Synthesizes a complete slice NALU header (start code + 2-byte NAL header
    /// + `slice_segment_header()` + `byte_alignment()`) into `writer`. This is
    /// the packed header submitted for `VA_ENC_PACKED_HEADER_SLICE`; the driver
    /// appends the coded slice data.
    ///
    /// `sps`/`pps` must be the active parameter sets — the many conditional
    /// fields of the header are gated on their values.
    pub fn synthesize(
        nalu_type: NaluType,
        hdr: &'n SliceHeader,
        sps: &Sps,
        pps: &Pps,
        writer: W,
        ep_enabled: bool,
    ) -> SynthesizerResult<()> {
        let mut s = Self { writer: NaluWriter::<W>::new(writer, ep_enabled), nalu: hdr };

        s.writer.write_header_hevc(nalu_type as u8, 0, 1)?;
        s.slice_segment_header(nalu_type, sps, pps)?;
        s.rbsp_trailing_bits()
    }

    /// `ref_pic_lists_modification()` — H.265 7.3.6.2.
    fn ref_pic_lists_modification(
        &mut self,
        rplm: &RefPicListModification,
        slice_type: SliceType,
        num_ref_idx_l0_active_minus1: u8,
        num_ref_idx_l1_active_minus1: u8,
        num_pic_total_curr: u32,
    ) -> SynthesizerResult<()> {
        let num_bits = ceil_log2(num_pic_total_curr);

        self.u(1, rplm.ref_pic_list_modification_flag_l0)?;
        if rplm.ref_pic_list_modification_flag_l0 {
            for i in 0..=num_ref_idx_l0_active_minus1 as usize {
                self.u(num_bits, rplm.list_entry_l0[i])?;
            }
        }

        if slice_type.is_b() {
            self.u(1, rplm.ref_pic_list_modification_flag_l1)?;
            if rplm.ref_pic_list_modification_flag_l1 {
                for i in 0..=num_ref_idx_l1_active_minus1 as usize {
                    self.u(num_bits, rplm.list_entry_l1[i])?;
                }
            }
        }

        Ok(())
    }

    /// `pred_weight_table()` — H.265 7.3.6.3.
    fn pred_weight_table(
        &mut self,
        pwt: &PredWeightTable,
        slice_type: SliceType,
        chroma_array_type: u8,
        num_ref_idx_l0_active_minus1: u8,
        num_ref_idx_l1_active_minus1: u8,
    ) -> SynthesizerResult<()> {
        self.ue(pwt.luma_log2_weight_denom)?;
        if chroma_array_type != 0 {
            self.se(pwt.delta_chroma_log2_weight_denom)?;
        }

        for i in 0..=num_ref_idx_l0_active_minus1 as usize {
            self.u(1, pwt.luma_weight_l0_flag[i])?;
        }
        if chroma_array_type != 0 {
            for i in 0..=num_ref_idx_l0_active_minus1 as usize {
                self.u(1, pwt.chroma_weight_l0_flag[i])?;
            }
        }

        for i in 0..=num_ref_idx_l0_active_minus1 as usize {
            if pwt.luma_weight_l0_flag[i] {
                self.se(pwt.delta_luma_weight_l0[i])?;
                self.se(pwt.luma_offset_l0[i])?;
            }
            if pwt.chroma_weight_l0_flag[i] {
                for j in 0..2 {
                    self.se(pwt.delta_chroma_weight_l0[i][j])?;
                    self.se(pwt.delta_chroma_offset_l0[i][j])?;
                }
            }
        }

        if slice_type.is_b() {
            for i in 0..=num_ref_idx_l1_active_minus1 as usize {
                self.u(1, pwt.luma_weight_l1_flag[i])?;
            }
            // §7.3.6.3: the L1 chroma_weight flag loop gates on ChromaArrayType,
            // matching the L0 loop above (not chroma_format_idc — they differ
            // when separate_colour_plane_flag == 1).
            if chroma_array_type != 0 {
                for i in 0..=num_ref_idx_l1_active_minus1 as usize {
                    self.u(1, pwt.chroma_weight_l1_flag[i])?;
                }
            }

            for i in 0..=num_ref_idx_l1_active_minus1 as usize {
                if pwt.luma_weight_l1_flag[i] {
                    self.se(pwt.delta_luma_weight_l1[i])?;
                    self.se(pwt.luma_offset_l1[i])?;
                }
                if pwt.chroma_weight_l1_flag[i] {
                    for j in 0..2 {
                        self.se(pwt.delta_chroma_weight_l1[i][j])?;
                        self.se(pwt.delta_chroma_offset_l1[i][j])?;
                    }
                }
            }
        }

        Ok(())
    }

    /// `slice_segment_header()` — H.265 7.3.6.1.
    fn slice_segment_header(
        &mut self,
        nalu_type: NaluType,
        sps: &Sps,
        pps: &Pps,
    ) -> SynthesizerResult<()> {
        let hdr = self.nalu;

        self.u(1, hdr.first_slice_segment_in_pic_flag)?;
        if nalu_type.is_irap() {
            self.u(1, hdr.no_output_of_prior_pics_flag)?;
        }
        self.ue(hdr.pic_parameter_set_id)?;

        if !hdr.first_slice_segment_in_pic_flag {
            if pps.dependent_slice_segments_enabled_flag {
                self.u(1, hdr.dependent_slice_segment_flag)?;
            }
            let num_bits = ceil_log2(sps.pic_size_in_ctbs_y);
            self.u(num_bits, hdr.segment_address)?;
        }

        if !hdr.dependent_slice_segment_flag {
            self.reserved_zero(usize::from(pps.num_extra_slice_header_bits))?;

            self.ue(hdr.type_ as u32)?;

            if pps.output_flag_present_flag {
                self.u(1, hdr.pic_output_flag)?;
            }
            if sps.separate_colour_plane_flag {
                self.u(2, hdr.colour_plane_id)?;
            }

            if !matches!(nalu_type, NaluType::IdrWRadl | NaluType::IdrNLp) {
                let num_bits = usize::from(sps.log2_max_pic_order_cnt_lsb_minus4 + 4);
                self.u(num_bits, hdr.pic_order_cnt_lsb)?;

                self.u(1, hdr.short_term_ref_pic_set_sps_flag)?;
                if !hdr.short_term_ref_pic_set_sps_flag {
                    self.short_term_ref_pic_set(
                        &hdr.short_term_ref_pic_set,
                        sps.num_short_term_ref_pic_sets,
                    )?;
                } else if sps.num_short_term_ref_pic_sets > 1 {
                    let num_bits = ceil_log2(u32::from(sps.num_short_term_ref_pic_sets));
                    self.u(num_bits, hdr.short_term_ref_pic_set_idx)?;
                }

                if sps.long_term_ref_pics_present_flag {
                    if sps.num_long_term_ref_pics_sps > 0 {
                        self.ue(hdr.num_long_term_sps)?;
                    }
                    self.ue(hdr.num_long_term_pics)?;

                    let num_lt = hdr.num_long_term_sps + hdr.num_long_term_pics;
                    for i in 0..num_lt as usize {
                        if i < hdr.num_long_term_sps as usize {
                            if sps.num_long_term_ref_pics_sps > 1 {
                                let num_bits =
                                    ceil_log2(u32::from(sps.num_long_term_ref_pics_sps));
                                self.u(num_bits, hdr.lt_idx_sps[i])?;
                            }
                        } else {
                            let num_bits = usize::from(sps.log2_max_pic_order_cnt_lsb_minus4) + 4;
                            self.u(num_bits, hdr.poc_lsb_lt[i])?;
                            self.u(1, hdr.used_by_curr_pic_lt[i])?;
                        }

                        self.u(1, hdr.delta_poc_msb_present_flag[i])?;
                        if hdr.delta_poc_msb_present_flag[i] {
                            // The parser stores the accumulated DeltaPocMsbCycleLt
                            // (eq. 7-52): for i != 0 && i != num_long_term_sps it
                            // adds the immediately-preceding element unconditionally.
                            // Invert that to recover the coded syntax element.
                            let coded = if i != 0 && i != hdr.num_long_term_sps as usize {
                                hdr.delta_poc_msb_cycle_lt[i] - hdr.delta_poc_msb_cycle_lt[i - 1]
                            } else {
                                hdr.delta_poc_msb_cycle_lt[i]
                            };
                            self.ue(coded)?;
                        }
                    }
                }

                if sps.temporal_mvp_enabled_flag {
                    self.u(1, hdr.temporal_mvp_enabled_flag)?;
                }
            }

            if sps.sample_adaptive_offset_enabled_flag {
                self.u(1, hdr.sao_luma_flag)?;
                if sps.chroma_array_type != 0 {
                    self.u(1, hdr.sao_chroma_flag)?;
                }
            }

            if hdr.type_.is_p() || hdr.type_.is_b() {
                self.u(1, hdr.num_ref_idx_active_override_flag)?;

                let (num_ref_idx_l0_active_minus1, num_ref_idx_l1_active_minus1) =
                    if hdr.num_ref_idx_active_override_flag {
                        self.ue(hdr.num_ref_idx_l0_active_minus1)?;
                        if hdr.type_.is_b() {
                            self.ue(hdr.num_ref_idx_l1_active_minus1)?;
                        }
                        (hdr.num_ref_idx_l0_active_minus1, hdr.num_ref_idx_l1_active_minus1)
                    } else {
                        (
                            pps.num_ref_idx_l0_default_active_minus1,
                            pps.num_ref_idx_l1_default_active_minus1,
                        )
                    };

                // NumPicTotalCurr (eq. 7-57).
                let rps = if hdr.short_term_ref_pic_set_sps_flag {
                    &sps.short_term_ref_pic_set[usize::from(hdr.curr_rps_idx)]
                } else {
                    &hdr.short_term_ref_pic_set
                };
                let mut num_pic_total_curr = 0u32;
                for i in 0..rps.num_negative_pics as usize {
                    num_pic_total_curr += u32::from(rps.used_by_curr_pic_s0[i]);
                }
                for i in 0..rps.num_positive_pics as usize {
                    num_pic_total_curr += u32::from(rps.used_by_curr_pic_s1[i]);
                }
                for i in 0..(hdr.num_long_term_sps + hdr.num_long_term_pics) as usize {
                    num_pic_total_curr += u32::from(hdr.used_by_curr_pic_lt[i]);
                }
                if pps.scc_extension.curr_pic_ref_enabled_flag {
                    num_pic_total_curr += 1;
                }

                if pps.lists_modification_present_flag && num_pic_total_curr > 1 {
                    self.ref_pic_lists_modification(
                        &hdr.ref_pic_list_modification,
                        hdr.type_,
                        num_ref_idx_l0_active_minus1,
                        num_ref_idx_l1_active_minus1,
                        num_pic_total_curr,
                    )?;
                }

                if hdr.type_.is_b() {
                    self.u(1, hdr.mvd_l1_zero_flag)?;
                }
                if pps.cabac_init_present_flag {
                    self.u(1, hdr.cabac_init_flag)?;
                }

                if hdr.temporal_mvp_enabled_flag {
                    if hdr.type_.is_b() {
                        self.u(1, hdr.collocated_from_l0_flag)?;
                    }
                    if (hdr.collocated_from_l0_flag && num_ref_idx_l0_active_minus1 > 0)
                        || (!hdr.collocated_from_l0_flag && num_ref_idx_l1_active_minus1 > 0)
                    {
                        self.ue(hdr.collocated_ref_idx)?;
                    }
                }

                if (pps.weighted_pred_flag && hdr.type_.is_p())
                    || (pps.weighted_bipred_flag && hdr.type_.is_b())
                {
                    self.pred_weight_table(
                        &hdr.pred_weight_table,
                        hdr.type_,
                        sps.chroma_array_type,
                        num_ref_idx_l0_active_minus1,
                        num_ref_idx_l1_active_minus1,
                    )?;
                }

                self.ue(hdr.five_minus_max_num_merge_cand)?;

                if sps.scc_extension.motion_vector_resolution_control_idc == 2 {
                    self.u(1, hdr.use_integer_mv_flag)?;
                }
            }

            self.se(hdr.qp_delta)?;

            if pps.slice_chroma_qp_offsets_present_flag {
                self.se(hdr.cb_qp_offset)?;
                self.se(hdr.cr_qp_offset)?;
            }
            if pps.scc_extension.slice_act_qp_offsets_present_flag {
                self.se(hdr.slice_act_y_qp_offset)?;
                self.se(hdr.slice_act_cb_qp_offset)?;
                self.se(hdr.slice_act_cr_qp_offset)?;
            }
            if pps.range_extension.chroma_qp_offset_list_enabled_flag {
                self.u(1, hdr.cu_chroma_qp_offset_enabled_flag)?;
            }

            if pps.deblocking_filter_override_enabled_flag {
                self.u(1, hdr.deblocking_filter_override_flag)?;
            }
            if hdr.deblocking_filter_override_flag {
                self.u(1, hdr.deblocking_filter_disabled_flag)?;
                if !hdr.deblocking_filter_disabled_flag {
                    self.se(hdr.beta_offset_div2)?;
                    self.se(hdr.tc_offset_div2)?;
                }
            }

            if pps.loop_filter_across_slices_enabled_flag
                && (hdr.sao_luma_flag
                    || hdr.sao_chroma_flag
                    || !hdr.deblocking_filter_disabled_flag)
            {
                self.u(1, hdr.loop_filter_across_slices_enabled_flag)?;
            }
        }

        if pps.tiles_enabled_flag || pps.entropy_coding_sync_enabled_flag {
            self.ue(hdr.num_entry_point_offsets)?;
            if hdr.num_entry_point_offsets > 0 {
                self.ue(hdr.offset_len_minus1)?;
                for i in 0..hdr.num_entry_point_offsets as usize {
                    self.u(
                        usize::from(hdr.offset_len_minus1 + 1),
                        hdr.entry_point_offset_minus1[i],
                    )?;
                }
            }
        }

        if pps.slice_segment_header_extension_present_flag {
            // slice_segment_header_extension_length — none emitted.
            self.ue(0u32)?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;
    use std::rc::Rc;

    use super::Synthesizer;
    use crate::codec::h264::nalu::Nalu;
    use crate::codec::h265::parser::HrdParams;
    use crate::codec::h265::parser::Level;
    use crate::codec::h265::parser::NaluHeader;
    use crate::codec::h265::parser::NaluType;
    use crate::codec::h265::parser::Parser;
    use crate::codec::h265::parser::Pps;
    use crate::codec::h265::parser::PpsRangeExtension;
    use crate::codec::h265::parser::PpsSccExtension;
    use crate::codec::h265::parser::ProfileTierLevel;
    use crate::codec::h265::parser::ScalingLists;
    use crate::codec::h265::parser::ShortTermRefPicSet;
    use crate::codec::h265::parser::SliceHeader;
    use crate::codec::h265::parser::SliceType;
    use crate::codec::h265::parser::Sps;
    use crate::codec::h265::parser::VuiParams;
    use crate::codec::h265::parser::Vps;

    /// Main-profile `profile_tier_level` (single layer).
    fn main_ptl() -> ProfileTierLevel {
        let mut compat = [false; 32];
        compat[1] = true;
        ProfileTierLevel {
            general_profile_idc: 1,
            general_profile_compatibility_flag: compat,
            general_progressive_source_flag: true,
            general_frame_only_constraint_flag: true,
            general_level_idc: Level::L4,
            ..Default::default()
        }
    }

    /// Main10-profile `profile_tier_level` (single layer).
    fn main10_ptl() -> ProfileTierLevel {
        let mut compat = [false; 32];
        compat[1] = true;
        compat[2] = true;
        ProfileTierLevel {
            general_profile_idc: 2,
            general_profile_compatibility_flag: compat,
            general_progressive_source_flag: true,
            general_frame_only_constraint_flag: true,
            general_level_idc: Level::L4,
            ..Default::default()
        }
    }

    /// A LowDelay-shaped SPS: 4:2:0, single sub-layer, CTB 64, no RPS in the
    /// SPS (P slices carry their own trivial RPS).
    fn make_sps(profile_idc: u8, bit_depth_minus8: u8) -> Sps {
        let ptl = if profile_idc == 2 { main10_ptl() } else { main_ptl() };
        let mut max_dpb = [0u8; 7];
        max_dpb[0] = 4;
        Sps {
            video_parameter_set_id: 0,
            max_sub_layers_minus1: 0,
            temporal_id_nesting_flag: true,
            profile_tier_level: ptl,
            seq_parameter_set_id: 0,
            chroma_format_idc: 1,
            pic_width_in_luma_samples: 320,
            pic_height_in_luma_samples: 240,
            bit_depth_luma_minus8: bit_depth_minus8,
            bit_depth_chroma_minus8: bit_depth_minus8,
            log2_max_pic_order_cnt_lsb_minus4: 4,
            sub_layer_ordering_info_present_flag: true,
            max_dec_pic_buffering_minus1: max_dpb,
            log2_min_luma_coding_block_size_minus3: 0,
            log2_diff_max_min_luma_coding_block_size: 3,
            log2_min_luma_transform_block_size_minus2: 0,
            log2_diff_max_min_luma_transform_block_size: 3,
            sample_adaptive_offset_enabled_flag: true,
            num_short_term_ref_pic_sets: 0,
            temporal_mvp_enabled_flag: true,
            ..Default::default()
        }
    }

    /// A minimal LowDelay PPS referencing `sps`.
    fn make_pps(sps: Rc<Sps>) -> Pps {
        Pps {
            pic_parameter_set_id: 0,
            seq_parameter_set_id: 0,
            dependent_slice_segments_enabled_flag: false,
            output_flag_present_flag: false,
            num_extra_slice_header_bits: 0,
            sign_data_hiding_enabled_flag: false,
            cabac_init_present_flag: false,
            num_ref_idx_l0_default_active_minus1: 0,
            num_ref_idx_l1_default_active_minus1: 0,
            init_qp_minus26: 0,
            constrained_intra_pred_flag: false,
            transform_skip_enabled_flag: false,
            cu_qp_delta_enabled_flag: true,
            diff_cu_qp_delta_depth: 0,
            cb_qp_offset: 0,
            cr_qp_offset: 0,
            slice_chroma_qp_offsets_present_flag: false,
            weighted_pred_flag: false,
            weighted_bipred_flag: false,
            transquant_bypass_enabled_flag: false,
            tiles_enabled_flag: false,
            entropy_coding_sync_enabled_flag: false,
            num_tile_columns_minus1: 0,
            num_tile_rows_minus1: 0,
            uniform_spacing_flag: true,
            column_width_minus1: [0; 19],
            row_height_minus1: [0; 21],
            loop_filter_across_tiles_enabled_flag: true,
            loop_filter_across_slices_enabled_flag: true,
            deblocking_filter_control_present_flag: false,
            deblocking_filter_override_enabled_flag: false,
            deblocking_filter_disabled_flag: false,
            beta_offset_div2: 0,
            tc_offset_div2: 0,
            scaling_list_data_present_flag: false,
            scaling_list: ScalingLists::default(),
            lists_modification_present_flag: false,
            log2_parallel_merge_level_minus2: 0,
            slice_segment_header_extension_present_flag: false,
            extension_present_flag: false,
            range_extension_flag: false,
            range_extension: PpsRangeExtension::default(),
            scc_extension_flag: false,
            scc_extension: PpsSccExtension::default(),
            qp_bd_offset_y: 0,
            sps,
        }
    }

    fn make_vps() -> Vps {
        let mut max_dpb = [0u32; 7];
        max_dpb[0] = 4;
        Vps {
            video_parameter_set_id: 0,
            base_layer_internal_flag: true,
            base_layer_available_flag: true,
            max_layers_minus1: 0,
            max_sub_layers_minus1: 0,
            temporal_id_nesting_flag: true,
            profile_tier_level: main_ptl(),
            sub_layer_ordering_info_present_flag: true,
            max_dec_pic_buffering_minus1: max_dpb,
            max_layer_id: 0,
            num_layer_sets_minus1: 0,
            timing_info_present_flag: false,
            extension_flag: false,
            ..Default::default()
        }
    }

    fn synth_vps(vps: &Vps) -> Vec<u8> {
        let mut buf = Vec::new();
        Synthesizer::<'_, Vps, _>::synthesize(vps, &mut buf, true).unwrap();
        buf
    }

    fn synth_sps(sps: &Sps) -> Vec<u8> {
        let mut buf = Vec::new();
        Synthesizer::<'_, Sps, _>::synthesize(sps, &mut buf, true).unwrap();
        buf
    }

    fn synth_pps(pps: &Pps) -> Vec<u8> {
        let mut buf = Vec::new();
        Synthesizer::<'_, Pps, _>::synthesize(pps, &mut buf, true).unwrap();
        buf
    }

    fn parse_vps(buf: &[u8]) -> Vps {
        let mut cursor = Cursor::new(buf);
        let nalu = Nalu::<NaluHeader>::next(&mut cursor).unwrap();
        assert_eq!(nalu.header.type_, NaluType::VpsNut);
        assert_eq!(nalu.header.nuh_layer_id, 0);
        assert_eq!(nalu.header.nuh_temporal_id_plus1, 1);
        let mut parser = Parser::default();
        (*parser.parse_vps(&nalu).unwrap()).clone()
    }

    fn parse_sps(buf: &[u8]) -> Sps {
        let mut cursor = Cursor::new(buf);
        let nalu = Nalu::<NaluHeader>::next(&mut cursor).unwrap();
        assert_eq!(nalu.header.type_, NaluType::SpsNut);
        let mut parser = Parser::default();
        (*parser.parse_sps(&nalu).unwrap()).clone()
    }

    /// Registers `sps` then parses `pps`, returning the parsed PPS.
    fn parse_pps(sps_buf: &[u8], pps_buf: &[u8]) -> Pps {
        let mut parser = Parser::default();

        let mut cursor = Cursor::new(sps_buf);
        let sps_nalu = Nalu::<NaluHeader>::next(&mut cursor).unwrap();
        parser.parse_sps(&sps_nalu).unwrap();

        let mut cursor = Cursor::new(pps_buf);
        let pps_nalu = Nalu::<NaluHeader>::next(&mut cursor).unwrap();
        assert_eq!(pps_nalu.header.type_, NaluType::PpsNut);
        (*parser.parse_pps(&pps_nalu).unwrap()).clone()
    }

    /// Registers the SPS+PPS, returning a `Parser` and clones of the active
    /// parameter sets to drive slice-header synthesis.
    fn parser_with_sps_pps() -> (Parser, Rc<Sps>, Rc<Pps>) {
        let sps = make_sps(1, 0);
        let sps_buf = synth_sps(&sps);
        let pps = make_pps(Rc::new(sps));
        let pps_buf = synth_pps(&pps);

        let mut parser = Parser::default();
        let mut cursor = Cursor::new(&sps_buf[..]);
        let sps_nalu = Nalu::<NaluHeader>::next(&mut cursor).unwrap();
        parser.parse_sps(&sps_nalu).unwrap();
        let mut cursor = Cursor::new(&pps_buf[..]);
        let pps_nalu = Nalu::<NaluHeader>::next(&mut cursor).unwrap();
        parser.parse_pps(&pps_nalu).unwrap();

        let sps = parser.get_sps(0).unwrap().clone();
        let pps = parser.get_pps(0).unwrap().clone();
        (parser, sps, pps)
    }

    #[test]
    fn synthesize_vps() {
        let vps = make_vps();
        let buf1 = synth_vps(&vps);
        let parsed1 = parse_vps(&buf1);

        assert_eq!(parsed1.video_parameter_set_id, 0);
        assert!(parsed1.base_layer_internal_flag);
        assert!(parsed1.base_layer_available_flag);
        assert_eq!(parsed1.max_sub_layers_minus1, 0);
        assert!(parsed1.temporal_id_nesting_flag);
        assert_eq!(parsed1.profile_tier_level.general_profile_idc, 1);
        assert_eq!(parsed1.profile_tier_level.general_level_idc, Level::L4);
        assert_eq!(parsed1.max_dec_pic_buffering_minus1[0], 4);
        assert!(!parsed1.timing_info_present_flag);

        let buf2 = synth_vps(&parsed1);
        assert_eq!(buf1, buf2, "VPS byte-idempotence");
        assert_eq!(parsed1, parse_vps(&buf2), "VPS struct equality");
    }

    #[test]
    fn synthesize_sps_main() {
        let mut sps = make_sps(1, 0);
        // Exercise the conformance-window path.
        sps.conformance_window_flag = true;
        sps.conf_win_bottom_offset = 1;

        let buf1 = synth_sps(&sps);
        let parsed1 = parse_sps(&buf1);

        assert_eq!(parsed1.profile_tier_level.general_profile_idc, 1);
        assert_eq!(parsed1.profile_tier_level.general_level_idc, Level::L4);
        assert_eq!(parsed1.chroma_format_idc, 1);
        assert_eq!(parsed1.bit_depth_luma_minus8, 0);
        assert_eq!(parsed1.bit_depth_chroma_minus8, 0);
        assert_eq!(parsed1.pic_width_in_luma_samples, 320);
        assert_eq!(parsed1.pic_height_in_luma_samples, 240);
        assert!(parsed1.conformance_window_flag);
        assert_eq!(parsed1.conf_win_bottom_offset, 1);
        assert!(parsed1.sample_adaptive_offset_enabled_flag);
        assert!(parsed1.temporal_mvp_enabled_flag);
        assert_eq!(parsed1.num_short_term_ref_pic_sets, 0);
        assert!(parsed1.vps.is_none());

        let buf2 = synth_sps(&parsed1);
        assert_eq!(buf1, buf2, "SPS byte-idempotence");
        assert_eq!(parsed1, parse_sps(&buf2), "SPS struct equality");
    }

    #[test]
    fn synthesize_sps_main10() {
        let sps = make_sps(2, 2);
        let buf1 = synth_sps(&sps);
        let parsed1 = parse_sps(&buf1);

        assert_eq!(parsed1.profile_tier_level.general_profile_idc, 2);
        assert!(parsed1.profile_tier_level.general_profile_compatibility_flag[2]);
        assert_eq!(parsed1.bit_depth_luma_minus8, 2);
        assert_eq!(parsed1.bit_depth_chroma_minus8, 2);

        let buf2 = synth_sps(&parsed1);
        assert_eq!(buf1, buf2, "Main10 SPS byte-idempotence");
        assert_eq!(parsed1, parse_sps(&buf2), "Main10 SPS struct equality");
    }

    #[test]
    fn synthesize_sps_scaling_lists() {
        let mut sps = make_sps(1, 0);
        sps.scaling_list_enabled_flag = true;
        sps.scaling_list_data_present_flag = true;
        sps.scaling_list = ScalingLists {
            scaling_list_4x4: [[16u8; 16]; 6],
            scaling_list_8x8: [[16u8; 64]; 6],
            scaling_list_16x16: [[16u8; 64]; 6],
            scaling_list_32x32: [[16u8; 64]; 6],
            scaling_list_dc_coef_minus8_16x16: [8i16; 6],
            scaling_list_dc_coef_minus8_32x32: [8i16; 6],
        };

        let buf1 = synth_sps(&sps);
        let parsed1 = parse_sps(&buf1);

        assert!(parsed1.scaling_list_enabled_flag);
        assert!(parsed1.scaling_list_data_present_flag);
        assert_eq!(parsed1.scaling_list.scaling_list_4x4, [[16u8; 16]; 6]);
        assert_eq!(parsed1.scaling_list.scaling_list_8x8, [[16u8; 64]; 6]);
        assert_eq!(parsed1.scaling_list.scaling_list_16x16, [[16u8; 64]; 6]);
        // 32x32 codes only matrix_id 0 and 3.
        assert_eq!(parsed1.scaling_list.scaling_list_32x32[0], [16u8; 64]);
        assert_eq!(parsed1.scaling_list.scaling_list_32x32[3], [16u8; 64]);
        assert_eq!(parsed1.scaling_list.scaling_list_dc_coef_minus8_16x16, [8i16; 6]);

        let buf2 = synth_sps(&parsed1);
        assert_eq!(buf1, buf2, "scaling-list SPS byte-idempotence");
        assert_eq!(parsed1, parse_sps(&buf2), "scaling-list SPS struct equality");
    }

    #[test]
    fn synthesize_sps_vui_hrd() {
        let mut sps = make_sps(1, 0);
        sps.vui_parameters_present_flag = true;

        let mut hrd = HrdParams::default();
        hrd.fixed_pic_rate_general_flag[0] = true;
        hrd.low_delay_hrd_flag[0] = false;
        hrd.cpb_cnt_minus1[0] = 0;

        let mut vui = VuiParams::default();
        vui.video_signal_type_present_flag = true;
        vui.video_format = 5;
        vui.video_full_range_flag = true;
        vui.colour_description_present_flag = true;
        vui.colour_primaries = 9;
        vui.transfer_characteristics = 16;
        vui.matrix_coeffs = 9;
        vui.timing_info_present_flag = true;
        vui.num_units_in_tick = 1000;
        vui.time_scale = 60000;
        vui.hrd_parameters_present_flag = true;
        vui.hrd = hrd;
        sps.vui_parameters = vui;

        let buf1 = synth_sps(&sps);
        let parsed1 = parse_sps(&buf1);

        assert!(parsed1.vui_parameters_present_flag);
        assert!(parsed1.vui_parameters.video_signal_type_present_flag);
        assert!(parsed1.vui_parameters.video_full_range_flag);
        assert_eq!(parsed1.vui_parameters.colour_primaries, 9);
        assert_eq!(parsed1.vui_parameters.transfer_characteristics, 16);
        assert_eq!(parsed1.vui_parameters.matrix_coeffs, 9);
        assert!(parsed1.vui_parameters.timing_info_present_flag);
        assert_eq!(parsed1.vui_parameters.num_units_in_tick, 1000);
        assert_eq!(parsed1.vui_parameters.time_scale, 60000);
        assert!(parsed1.vui_parameters.hrd_parameters_present_flag);
        // §E.2.2: fixed_pic_rate_general_flag == 1 is a spec-compliant HRD in
        // which fixed_pic_rate_within_cvs_flag is not coded but inferred to 1.
        assert!(parsed1.vui_parameters.hrd.fixed_pic_rate_general_flag[0]);
        assert!(
            parsed1.vui_parameters.hrd.fixed_pic_rate_within_cvs_flag[0],
            "within_cvs inferred to 1 when general_flag == 1"
        );
        assert_eq!(parsed1.vui_parameters.hrd.elemental_duration_in_tc_minus1[0], 0);

        let buf2 = synth_sps(&parsed1);
        assert_eq!(buf1, buf2, "VUI/HRD SPS byte-idempotence");
        assert_eq!(parsed1, parse_sps(&buf2), "VUI/HRD SPS struct equality");
    }

    #[test]
    fn synthesize_pps() {
        let sps = make_sps(1, 0);
        let sps_buf = synth_sps(&sps);
        let pps = make_pps(Rc::new(sps));
        let pps_buf1 = synth_pps(&pps);

        let parsed1 = parse_pps(&sps_buf, &pps_buf1);
        assert_eq!(parsed1.pic_parameter_set_id, 0);
        assert_eq!(parsed1.seq_parameter_set_id, 0);
        assert!(parsed1.cu_qp_delta_enabled_flag);
        assert!(!parsed1.tiles_enabled_flag);
        assert!(!parsed1.weighted_pred_flag);
        assert_eq!(parsed1.init_qp_minus26, 0);
        assert!(parsed1.loop_filter_across_slices_enabled_flag);

        let pps_buf2 = synth_pps(&parsed1);
        assert_eq!(pps_buf1, pps_buf2, "PPS byte-idempotence");
        assert_eq!(parsed1, parse_pps(&sps_buf, &pps_buf2), "PPS struct equality");
    }

    #[test]
    fn synthesize_idr_slice_header() {
        let (mut parser, sps, pps) = parser_with_sps_pps();

        let hdr = SliceHeader {
            first_slice_segment_in_pic_flag: true,
            no_output_of_prior_pics_flag: false,
            pic_parameter_set_id: 0,
            type_: SliceType::I,
            sao_luma_flag: true,
            sao_chroma_flag: true,
            qp_delta: 4,
            loop_filter_across_slices_enabled_flag: true,
            ..Default::default()
        };

        let mut buf1 = Vec::new();
        Synthesizer::<'_, SliceHeader, _>::synthesize(
            NaluType::IdrWRadl,
            &hdr,
            &sps,
            &pps,
            &mut buf1,
            true,
        )
        .unwrap();

        let mut cursor = Cursor::new(&buf1[..]);
        let nalu = Nalu::<NaluHeader>::next(&mut cursor).unwrap();
        assert_eq!(nalu.header.type_, NaluType::IdrWRadl);
        let parsed1 = parser.parse_slice_header(nalu).unwrap().header;

        assert!(parsed1.first_slice_segment_in_pic_flag);
        assert!(!parsed1.no_output_of_prior_pics_flag);
        assert_eq!(parsed1.type_, SliceType::I);
        assert!(parsed1.sao_luma_flag);
        assert!(parsed1.sao_chroma_flag);
        assert_eq!(parsed1.qp_delta, 4);

        let mut buf2 = Vec::new();
        Synthesizer::<'_, SliceHeader, _>::synthesize(
            NaluType::IdrWRadl,
            &parsed1,
            &sps,
            &pps,
            &mut buf2,
            true,
        )
        .unwrap();
        assert_eq!(buf1, buf2, "IDR slice byte-idempotence");

        let mut cursor = Cursor::new(&buf2[..]);
        let nalu2 = Nalu::<NaluHeader>::next(&mut cursor).unwrap();
        let parsed2 = parser.parse_slice_header(nalu2).unwrap().header;
        assert_eq!(parsed1, parsed2, "IDR slice struct equality");
    }

    #[test]
    fn synthesize_p_slice_header() {
        let (mut parser, sps, pps) = parser_with_sps_pps();

        // Trivial LowDelay RPS: one negative ref, delta_poc_s0_minus1 == 0.
        let mut rps = ShortTermRefPicSet::default();
        rps.num_negative_pics = 1;
        rps.num_positive_pics = 0;
        rps.delta_poc_s0[0] = -1;
        rps.used_by_curr_pic_s0[0] = true;
        rps.num_delta_pocs = 1;

        let hdr = SliceHeader {
            first_slice_segment_in_pic_flag: true,
            pic_parameter_set_id: 0,
            type_: SliceType::P,
            pic_order_cnt_lsb: 1,
            short_term_ref_pic_set_sps_flag: false,
            short_term_ref_pic_set: rps,
            temporal_mvp_enabled_flag: true,
            sao_luma_flag: true,
            sao_chroma_flag: true,
            num_ref_idx_active_override_flag: false,
            five_minus_max_num_merge_cand: 3,
            qp_delta: 4,
            loop_filter_across_slices_enabled_flag: true,
            ..Default::default()
        };

        let mut buf1 = Vec::new();
        Synthesizer::<'_, SliceHeader, _>::synthesize(
            NaluType::TrailR,
            &hdr,
            &sps,
            &pps,
            &mut buf1,
            true,
        )
        .unwrap();

        let mut cursor = Cursor::new(&buf1[..]);
        let nalu = Nalu::<NaluHeader>::next(&mut cursor).unwrap();
        assert_eq!(nalu.header.type_, NaluType::TrailR);
        let parsed1 = parser.parse_slice_header(nalu).unwrap().header;

        assert_eq!(parsed1.type_, SliceType::P);
        assert_eq!(parsed1.pic_order_cnt_lsb, 1);
        assert!(!parsed1.short_term_ref_pic_set_sps_flag);
        assert_eq!(parsed1.short_term_ref_pic_set.num_negative_pics, 1);
        assert_eq!(parsed1.short_term_ref_pic_set.num_positive_pics, 0);
        assert_eq!(parsed1.short_term_ref_pic_set.delta_poc_s0[0], -1);
        assert!(parsed1.short_term_ref_pic_set.used_by_curr_pic_s0[0]);
        assert_eq!(parsed1.num_pic_total_curr, 1);
        assert_eq!(parsed1.five_minus_max_num_merge_cand, 3);
        assert_eq!(parsed1.qp_delta, 4);

        let mut buf2 = Vec::new();
        Synthesizer::<'_, SliceHeader, _>::synthesize(
            NaluType::TrailR,
            &parsed1,
            &sps,
            &pps,
            &mut buf2,
            true,
        )
        .unwrap();
        assert_eq!(buf1, buf2, "P slice byte-idempotence");

        let mut cursor = Cursor::new(&buf2[..]);
        let nalu2 = Nalu::<NaluHeader>::next(&mut cursor).unwrap();
        let parsed2 = parser.parse_slice_header(nalu2).unwrap().header;
        assert_eq!(parsed1, parsed2, "P slice struct equality");
    }

    /// Registers `sps` + `pps` in a fresh parser, returning it with the active
    /// parameter sets to drive slice-header synthesis.
    fn register(sps: &Sps, pps: &Pps) -> (Parser, Rc<Sps>, Rc<Pps>) {
        let sps_buf = synth_sps(sps);
        let pps_buf = synth_pps(pps);

        let mut parser = Parser::default();
        let mut cursor = Cursor::new(&sps_buf[..]);
        let sps_nalu = Nalu::<NaluHeader>::next(&mut cursor).unwrap();
        parser.parse_sps(&sps_nalu).unwrap();
        let mut cursor = Cursor::new(&pps_buf[..]);
        let pps_nalu = Nalu::<NaluHeader>::next(&mut cursor).unwrap();
        parser.parse_pps(&pps_nalu).unwrap();

        let sps = parser.get_sps(0).unwrap().clone();
        let pps = parser.get_pps(0).unwrap().clone();
        (parser, sps, pps)
    }

    /// Finding 5 — the sub-layer PTL profile block gates on
    /// `sub_layer_profile_present_flag[i]` (§7.3.3), not the level-present flag.
    /// Drives the case `profile_present = true, level_present = false`, in which
    /// the two gates disagree: only the corrected gate writes/reads the profile
    /// block, so `sub_layer_profile_idc[0]` survives the round-trip.
    #[test]
    fn synthesize_sps_sub_layer_ptl() {
        let mut sps = make_sps(1, 0);
        sps.max_sub_layers_minus1 = 1;
        sps.max_dec_pic_buffering_minus1[1] = 4;

        let ptl = &mut sps.profile_tier_level;
        ptl.sub_layer_profile_present_flag[0] = true;
        ptl.sub_layer_level_present_flag[0] = false;
        ptl.sub_layer_profile_idc[0] = 1;
        ptl.sub_layer_profile_compatibility_flag[0][1] = true;
        ptl.sub_layer_progressive_source_flag[0] = true;
        ptl.sub_layer_frame_only_constraint_flag[0] = true;

        let buf1 = synth_sps(&sps);
        let parsed1 = parse_sps(&buf1);

        let pptl = &parsed1.profile_tier_level;
        assert!(pptl.sub_layer_profile_present_flag[0]);
        assert!(!pptl.sub_layer_level_present_flag[0]);
        // Would be 0 (block skipped) if the gate still used level_present_flag.
        assert_eq!(pptl.sub_layer_profile_idc[0], 1);
        assert!(pptl.sub_layer_profile_compatibility_flag[0][1]);
        assert!(pptl.sub_layer_progressive_source_flag[0]);
        assert!(pptl.sub_layer_frame_only_constraint_flag[0]);

        let buf2 = synth_sps(&parsed1);
        assert_eq!(buf1, buf2, "sub-layer PTL SPS byte-idempotence");
        assert_eq!(parsed1, parse_sps(&buf2), "sub-layer PTL SPS struct equality");
    }

    /// Finding 2 — the writer inverts the parser's accumulated
    /// `DeltaPocMsbCycleLt` (eq. 7-52) using the immediately-preceding array
    /// element. Drives long-term refs straddling the `num_long_term_sps`
    /// boundary with a present/absent/present pattern, where the removed
    /// "last-present" tracker would have produced a wrong coded value.
    #[test]
    fn synthesize_p_slice_long_term_refs() {
        let mut sps = make_sps(1, 0);
        sps.long_term_ref_pics_present_flag = true;
        sps.num_long_term_ref_pics_sps = 1;
        sps.lt_ref_pic_poc_lsb_sps[0] = 8;
        sps.used_by_curr_pic_lt_sps_flag[0] = true;

        let pps = make_pps(Rc::new(sps.clone()));
        let (mut parser, sps, pps) = register(&sps, &pps);

        let mut rps = ShortTermRefPicSet::default();
        rps.num_negative_pics = 1;
        rps.num_positive_pics = 0;
        rps.delta_poc_s0[0] = -1;
        rps.used_by_curr_pic_s0[0] = true;
        rps.num_delta_pocs = 1;

        let mut hdr = SliceHeader {
            first_slice_segment_in_pic_flag: true,
            pic_parameter_set_id: 0,
            type_: SliceType::P,
            pic_order_cnt_lsb: 2,
            short_term_ref_pic_set_sps_flag: false,
            short_term_ref_pic_set: rps,
            num_long_term_sps: 1,
            num_long_term_pics: 2,
            temporal_mvp_enabled_flag: false,
            sao_luma_flag: true,
            sao_chroma_flag: true,
            num_ref_idx_active_override_flag: false,
            five_minus_max_num_merge_cand: 3,
            qp_delta: 4,
            loop_filter_across_slices_enabled_flag: true,
            ..Default::default()
        };
        // i == 0: SPS-based LT ref, present. Accumulated == coded (reset at i==0).
        hdr.used_by_curr_pic_lt[0] = true;
        hdr.delta_poc_msb_present_flag[0] = true;
        hdr.delta_poc_msb_cycle_lt[0] = 2;
        // i == 1: explicit LT ref, absent. i == num_long_term_sps → reset point.
        hdr.poc_lsb_lt[1] = 5;
        hdr.used_by_curr_pic_lt[1] = false;
        hdr.delta_poc_msb_present_flag[1] = false;
        // i == 2: explicit LT ref, present. Accumulated = coded(3) + arr[1](0).
        hdr.poc_lsb_lt[2] = 6;
        hdr.used_by_curr_pic_lt[2] = true;
        hdr.delta_poc_msb_present_flag[2] = true;
        hdr.delta_poc_msb_cycle_lt[2] = 3;

        let mut buf1 = Vec::new();
        Synthesizer::<'_, SliceHeader, _>::synthesize(
            NaluType::TrailR,
            &hdr,
            &sps,
            &pps,
            &mut buf1,
            true,
        )
        .unwrap();

        let mut cursor = Cursor::new(&buf1[..]);
        let nalu = Nalu::<NaluHeader>::next(&mut cursor).unwrap();
        let parsed1 = parser.parse_slice_header(nalu).unwrap().header;

        assert_eq!(parsed1.num_long_term_sps, 1);
        assert_eq!(parsed1.num_long_term_pics, 2);
        assert!(parsed1.delta_poc_msb_present_flag[0]);
        assert_eq!(parsed1.delta_poc_msb_cycle_lt[0], 2);
        assert!(!parsed1.delta_poc_msb_present_flag[1]);
        assert_eq!(parsed1.delta_poc_msb_cycle_lt[1], 0);
        assert!(parsed1.delta_poc_msb_present_flag[2]);
        // Distinguishes the fix: the old "last-present" tracker would code 1
        // here (3 - arr[0]) instead of 3, decoding back to 1, not 3.
        assert_eq!(parsed1.delta_poc_msb_cycle_lt[2], 3);

        let mut buf2 = Vec::new();
        Synthesizer::<'_, SliceHeader, _>::synthesize(
            NaluType::TrailR,
            &parsed1,
            &sps,
            &pps,
            &mut buf2,
            true,
        )
        .unwrap();
        assert_eq!(buf1, buf2, "LT-ref P slice byte-idempotence");

        let mut cursor = Cursor::new(&buf2[..]);
        let nalu2 = Nalu::<NaluHeader>::next(&mut cursor).unwrap();
        let parsed2 = parser.parse_slice_header(nalu2).unwrap().header;
        assert_eq!(parsed1, parsed2, "LT-ref P slice struct equality");
    }

    /// Finding 4 — the L1 `chroma_weight` flag loop gates on ChromaArrayType
    /// (§7.3.6.3), matching L0. Drives a weighted-bipred B slice with
    /// `separate_colour_plane_flag = 1` (chroma_format_idc = 3, ChromaArrayType
    /// = 0), the one config where the two gates disagree: the L1 chroma flags
    /// must be absent. A one-sided revert to `chroma_format_idc` on either the
    /// writer or the parser would desync this round-trip.
    #[test]
    fn synthesize_b_slice_pred_weight_separate_colour_plane() {
        let mut sps = make_sps(1, 0);
        sps.chroma_format_idc = 3;
        sps.separate_colour_plane_flag = true;
        sps.temporal_mvp_enabled_flag = false;

        let mut pps = make_pps(Rc::new(sps.clone()));
        pps.weighted_bipred_flag = true;

        let (mut parser, sps, pps) = register(&sps, &pps);
        assert_eq!(sps.chroma_array_type, 0);

        let mut rps = ShortTermRefPicSet::default();
        rps.num_negative_pics = 1;
        rps.num_positive_pics = 0;
        rps.delta_poc_s0[0] = -1;
        rps.used_by_curr_pic_s0[0] = true;
        rps.num_delta_pocs = 1;

        let mut hdr = SliceHeader {
            first_slice_segment_in_pic_flag: true,
            pic_parameter_set_id: 0,
            type_: SliceType::B,
            pic_order_cnt_lsb: 2,
            short_term_ref_pic_set_sps_flag: false,
            short_term_ref_pic_set: rps,
            colour_plane_id: 0,
            sao_luma_flag: true,
            num_ref_idx_active_override_flag: false,
            mvd_l1_zero_flag: false,
            five_minus_max_num_merge_cand: 3,
            qp_delta: 4,
            loop_filter_across_slices_enabled_flag: true,
            ..Default::default()
        };
        // Luma-only weights (ChromaArrayType == 0 ⇒ no chroma flags/weights).
        hdr.pred_weight_table.luma_weight_l0_flag[0] = true;
        hdr.pred_weight_table.delta_luma_weight_l0[0] = 1;
        hdr.pred_weight_table.luma_offset_l0[0] = 2;
        hdr.pred_weight_table.luma_weight_l1_flag[0] = true;
        hdr.pred_weight_table.delta_luma_weight_l1[0] = -1;
        hdr.pred_weight_table.luma_offset_l1[0] = 3;

        let mut buf1 = Vec::new();
        Synthesizer::<'_, SliceHeader, _>::synthesize(
            NaluType::TrailR,
            &hdr,
            &sps,
            &pps,
            &mut buf1,
            true,
        )
        .unwrap();

        let mut cursor = Cursor::new(&buf1[..]);
        let nalu = Nalu::<NaluHeader>::next(&mut cursor).unwrap();
        let parsed1 = parser.parse_slice_header(nalu).unwrap().header;

        assert_eq!(parsed1.type_, SliceType::B);
        assert!(parsed1.pred_weight_table.luma_weight_l0_flag[0]);
        assert!(parsed1.pred_weight_table.luma_weight_l1_flag[0]);
        assert_eq!(parsed1.pred_weight_table.delta_luma_weight_l1[0], -1);
        assert_eq!(parsed1.pred_weight_table.luma_offset_l1[0], 3);
        // ChromaArrayType == 0 ⇒ no L1 (or L0) chroma weight flags coded.
        assert!(!parsed1.pred_weight_table.chroma_weight_l1_flag[0]);
        assert!(!parsed1.pred_weight_table.chroma_weight_l0_flag[0]);

        let mut buf2 = Vec::new();
        Synthesizer::<'_, SliceHeader, _>::synthesize(
            NaluType::TrailR,
            &parsed1,
            &sps,
            &pps,
            &mut buf2,
            true,
        )
        .unwrap();
        assert_eq!(buf1, buf2, "separate-colour-plane B slice byte-idempotence");

        let mut cursor = Cursor::new(&buf2[..]);
        let nalu2 = Nalu::<NaluHeader>::next(&mut cursor).unwrap();
        let parsed2 = parser.parse_slice_header(nalu2).unwrap().header;
        assert_eq!(parsed1, parsed2, "separate-colour-plane B slice struct equality");
    }
}
