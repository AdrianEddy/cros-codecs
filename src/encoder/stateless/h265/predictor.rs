// Copyright 2024 The ChromiumOS Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

//! HEVC LowDelay-IPPP predictor: VPS/SPS/PPS field selection and per-frame
//! request construction. Mirrors [`crate::encoder::stateless::h264::predictor`].
//!
//! HEVC-specific decisions (Main, 8-bit 4:2:0, single slice / tile / sub-layer):
//! - **Profile-tier-level**: `general_profile_idc == 1` (Main), compat[1] set,
//!   progressive + frame-only, the config's level.
//! - **Coding-block sizes**: min CB `8` (`log2_min_luma_coding_block_size_minus3
//!   == 0`), CTB `32` (`log2_diff_max_min_luma_coding_block_size == 2`) — the
//!   a broadly supported default. Min TB `4`, max
//!   TB `32`, transform-hierarchy depth `2`.
//! - **`dec_pic_buf_mgr`**: `sps_max_dec_pic_buffering_minus1 == 1` (current + one
//!   reference), `max_num_reorder_pics == 0` (LowDelay, no reordering).
//! - **RPS**: the trivial LowDelay short-term RPS is carried *in the slice header*
//!   (`num_short_term_ref_pic_sets == 0` in the SPS): one negative reference at
//!   `delta_poc_s0 == -1`, `used_by_curr_pic_s0`.
//! - **POC**: `poc == counter` (resets to 0 at each IDR); `pic_order_cnt_lsb ==
//!   poc mod MaxPicOrderCntLsb`, sized so the GOP never wraps.
//! - **Off for simplicity/robustness (all standard-conformant)**: SAO,
//!   temporal-MVP, scaling lists, PCM, transform-skip, sign-data-hiding,
//!   weighted-prediction. AMP and CU-QP-delta (needed for CBR rate control) are on.

use std::rc::Rc;

use log::trace;

use crate::codec::h265::parser::Level;
use crate::codec::h265::parser::NaluType;
use crate::codec::h265::parser::Pps;
use crate::codec::h265::parser::Profile;
use crate::codec::h265::parser::ProfileTierLevel;
use crate::codec::h265::parser::ShortTermRefPicSet;
use crate::codec::h265::parser::SliceHeader;
use crate::codec::h265::parser::SliceType;
use crate::codec::h265::parser::Sps;
use crate::codec::h265::parser::VuiParams;
use crate::codec::h265::parser::Vps;
use crate::encoder::stateless::h265::BackendRequest;
use crate::encoder::stateless::h265::DpbEntry;
use crate::encoder::stateless::h265::DpbEntryMeta;
use crate::encoder::stateless::h265::EncoderConfig;
use crate::encoder::stateless::predictor::LowDelay;
use crate::encoder::stateless::predictor::LowDelayDelegate;
use crate::encoder::stateless::FrameMetadata;
use crate::encoder::EncodeError;
use crate::encoder::EncodeResult;
use crate::encoder::RateControl;
use crate::encoder::Tunings;

/// The slice QP is `26 + init_qp_minus26 + slice_qp_delta`; clamp `pic_init_qp`
/// into the 8-bit HEVC QP range `[1, 51]` (QP 0 is a pathological extreme the
/// config layer never requests).
pub(crate) const MIN_QP: u8 = 1;
pub(crate) const MAX_QP: u8 = 51;

/// Min luma coding-block size log2 minus 3 → MinCbSizeY = 8.
const LOG2_MIN_CB_SIZE_MINUS3: u8 = 0;
/// Max minus min luma coding-block size log2 → CtbSizeY = 32 (log2 = 3 + 2).
const LOG2_DIFF_MAX_MIN_CB_SIZE: u8 = 2;
const MIN_CB_LOG2: u32 = 3 + LOG2_MIN_CB_SIZE_MINUS3 as u32;
const CTB_LOG2: u32 = MIN_CB_LOG2 + LOG2_DIFF_MAX_MIN_CB_SIZE as u32;
/// CtbSizeY = 32.
const CTB_SIZE: u32 = 1 << CTB_LOG2;
/// MinCbSizeY = 8 — `pic_{width,height}_in_luma_samples` must be a multiple of it.
const MIN_CB_SIZE: u32 = 1 << MIN_CB_LOG2;

/// `(SubWidthC, SubHeightC)` for a `chroma_format_idc`:
/// 4:2:0 → (2, 2); 4:2:2 → (2, 1). Monochrome/4:4:4 are never emitted.
fn sub_wh_c(chroma_format_idc: u8) -> (u32, u32) {
    match chroma_format_idc {
        2 => (2, 1), // 4:2:2
        _ => (2, 2), // 4:2:0
    }
}

/// Round `x` up to a multiple of `a` (`a` a power of two ≥ 1).
fn align_up(x: u32, a: u32) -> u32 {
    x.div_ceil(a) * a
}

/// The smallest `log2_max_pic_order_cnt_lsb_minus4` (∈ `[0, 12]`) whose
/// `MaxPicOrderCntLsb = 2^(v+4)` covers `2 * limit`, so the POC never wraps
/// within a GOP (references stay a clean `delta_poc == -1` away).
fn log2_max_poc_lsb_minus4(limit: u16) -> u8 {
    let need = 2u32.saturating_mul(limit.max(1) as u32);
    let mut v = 0u8;
    while (1u32 << (v + 4)) < need && v < 12 {
        v += 1;
    }
    v
}

/// The `profile_tier_level` for a single-sub-layer stream of the given profile.
///
/// - `Main` / `Main10`: `general_profile_idc` 1 / 2, no explicit RExt
///   constraint block (the synthesizer emits the Main10 `[2]` reserved form).
/// - `RangeExtensions`: `general_profile_idc == 4` with the **Main 4:2:2 10**
///   general-constraint indicator flags: max_12bit,
///   max_10bit, max_422chroma and lower_bit_rate set, everything else clear —
///   the only RExt profile the encoder emits.
fn build_ptl(profile: Profile, level: Level) -> ProfileTierLevel {
    let idc = profile as u8;
    let mut compat = [false; 32];
    compat[idc as usize] = true;
    let mut ptl = ProfileTierLevel {
        general_profile_idc: idc,
        general_profile_compatibility_flag: compat,
        general_progressive_source_flag: true,
        general_frame_only_constraint_flag: true,
        general_level_idc: level,
        ..Default::default()
    };
    if profile == Profile::RangeExtensions {
        // Main 4:2:2 10 constraint flags.
        ptl.general_max_12bit_constraint_flag = true;
        ptl.general_max_10bit_constraint_flag = true;
        ptl.general_max_8bit_constraint_flag = false;
        ptl.general_max_422chroma_constraint_flag = true;
        ptl.general_max_420chroma_constraint_flag = false;
        ptl.general_max_monochrome_constraint_flag = false;
        ptl.general_intra_constraint_flag = false;
        ptl.general_one_picture_only_constraint_flag = false;
        ptl.general_lower_bit_rate_constraint_flag = true;
    }
    ptl
}

/// Bit depth (as `bit_depth_*_minus8`) and chroma format (`chroma_format_idc`)
/// for a general profile: `Main` → 8-bit 4:2:0, `Main10` → 10-bit 4:2:0,
/// `RangeExtensions` → 10-bit 4:2:2 (Main 4:2:2 10). Any other profile is
/// rejected earlier at `new_vaapi`; default to Main's 8-bit 4:2:0.
fn profile_format(profile: Profile) -> (u8, u8) {
    match profile {
        Profile::Main => (0, 1),
        Profile::Main10 => (2, 1),
        Profile::RangeExtensions => (2, 2),
        _ => (0, 1),
    }
}

/// Build the (VPS, SPS, PPS) triple for a Main 8-bit 4:2:0 LowDelay stream from
/// the encoder config + current tunings. Split out (rather than inlined into
/// [`LowDelayH265::new_sequence`]) so the unit tests can round-trip each set
/// through the synthesizer without a live encoder.
pub(crate) fn build_parameter_sets(
    config: &EncoderConfig,
    tunings: &Tunings,
    limit: u16,
) -> (Rc<Vps>, Rc<Sps>, Rc<Pps>) {
    let width = config.resolution.width;
    let height = config.resolution.height;
    // HEVC requires the coded dimensions to be a multiple of MinCbSizeY; pad up
    // and crop the difference back to the display size with a conformance window
    // (the H.264 `pic_*_in_mbs` + `frame_cropping` pattern, at 8-px granularity).
    let coded_w = align_up(width, MIN_CB_SIZE);
    let coded_h = align_up(height, MIN_CB_SIZE);

    let level = config.level;
    let ptl = build_ptl(config.profile, level);

    // Bit depth (Main → 8, Main10/Main422_10 → 10) and chroma format (4:2:0 vs
    // 4:2:2) are a function of the general profile. The conformance-window
    // offsets are in chroma sample units, so they use SubWidthC/SubHeightC.
    let (bit_depth_minus8, chroma_format_idc) = profile_format(config.profile);
    let (sub_width_c, sub_height_c) = sub_wh_c(chroma_format_idc);

    // dec_pic_buf_mgr: current + one reference ⇒ buffering of 2 (minus1 = 1),
    // no reordering (LowDelay), no latency bound.
    let mut max_dec_pic_buffering_minus1 = [0u8; 7];
    max_dec_pic_buffering_minus1[0] = 1;
    let max_num_reorder_pics = [0u8; 7];
    let max_latency_increase_plus1 = [0u8; 7];

    let log2_max_poc_lsb_minus4 = log2_max_poc_lsb_minus4(limit);

    let pic_w_ctbs = coded_w.div_ceil(CTB_SIZE);
    let pic_h_ctbs = coded_h.div_ceil(CTB_SIZE);

    let conformance_window_flag = coded_w != width || coded_h != height;

    // An optional CICP colour description is carried in the VUI. Without one,
    // the VUI is omitted entirely.
    let (vui_parameters_present_flag, vui_parameters) = match config.color {
        Some(c) => (
            true,
            VuiParams {
                video_signal_type_present_flag: true,
                video_full_range_flag: c.full_range,
                colour_description_present_flag: true,
                colour_primaries: c.primaries as u32,
                transfer_characteristics: c.transfer as u32,
                matrix_coeffs: c.matrix as u32,
                ..Default::default()
            },
        ),
        None => (false, VuiParams::default()),
    };

    let sps = Sps {
        video_parameter_set_id: 0,
        max_sub_layers_minus1: 0,
        temporal_id_nesting_flag: true,
        profile_tier_level: ptl.clone(),
        seq_parameter_set_id: 0,
        chroma_format_idc,
        separate_colour_plane_flag: false,
        pic_width_in_luma_samples: coded_w as u16,
        pic_height_in_luma_samples: coded_h as u16,
        conformance_window_flag,
        conf_win_left_offset: 0,
        conf_win_right_offset: (coded_w - width) / sub_width_c,
        conf_win_top_offset: 0,
        conf_win_bottom_offset: (coded_h - height) / sub_height_c,
        bit_depth_luma_minus8: bit_depth_minus8,
        bit_depth_chroma_minus8: bit_depth_minus8,
        log2_max_pic_order_cnt_lsb_minus4: log2_max_poc_lsb_minus4,
        sub_layer_ordering_info_present_flag: true,
        max_dec_pic_buffering_minus1,
        max_num_reorder_pics,
        max_latency_increase_plus1,
        log2_min_luma_coding_block_size_minus3: LOG2_MIN_CB_SIZE_MINUS3,
        log2_diff_max_min_luma_coding_block_size: LOG2_DIFF_MAX_MIN_CB_SIZE,
        log2_min_luma_transform_block_size_minus2: 0,
        log2_diff_max_min_luma_transform_block_size: 3,
        max_transform_hierarchy_depth_inter: 2,
        max_transform_hierarchy_depth_intra: 2,
        scaling_list_enabled_flag: false,
        scaling_list_data_present_flag: false,
        amp_enabled_flag: true,
        sample_adaptive_offset_enabled_flag: false,
        pcm_enabled_flag: false,
        // No RPS in the SPS — the P slices carry their own trivial in-header RPS.
        num_short_term_ref_pic_sets: 0,
        short_term_ref_pic_set: vec![],
        long_term_ref_pics_present_flag: false,
        temporal_mvp_enabled_flag: false,
        strong_intra_smoothing_enabled_flag: false,
        vui_parameters_present_flag,
        vui_parameters,
        extension_present_flag: false,
        // Computed fields the synthesizer / backend read.
        chroma_array_type: chroma_format_idc,
        min_cb_log2_size_y: MIN_CB_LOG2,
        ctb_log2_size_y: CTB_LOG2,
        ctb_size_y: CTB_SIZE,
        pic_width_in_ctbs_y: pic_w_ctbs,
        pic_height_in_ctbs_y: pic_h_ctbs,
        pic_size_in_ctbs_y: pic_w_ctbs * pic_h_ctbs,
        pic_size_in_samples_y: coded_w * coded_h,
        max_tb_log2_size_y: 5,
        vps: None,
        ..Default::default()
    };
    let sps = Rc::new(sps);

    let init_qp = pick_init_qp(tunings);

    let pps = Pps {
        pic_parameter_set_id: 0,
        seq_parameter_set_id: 0,
        dependent_slice_segments_enabled_flag: false,
        output_flag_present_flag: false,
        num_extra_slice_header_bits: 0,
        sign_data_hiding_enabled_flag: false,
        cabac_init_present_flag: false,
        num_ref_idx_l0_default_active_minus1: 0,
        num_ref_idx_l1_default_active_minus1: 0,
        init_qp_minus26: init_qp as i8 - 26,
        constrained_intra_pred_flag: false,
        transform_skip_enabled_flag: false,
        // CU-level QP delta is needed for the CBR rate controller to vary QP.
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
        scaling_list: Default::default(),
        lists_modification_present_flag: false,
        log2_parallel_merge_level_minus2: 0,
        slice_segment_header_extension_present_flag: false,
        extension_present_flag: false,
        range_extension_flag: false,
        range_extension: Default::default(),
        scc_extension_flag: false,
        scc_extension: Default::default(),
        qp_bd_offset_y: 0,
        sps: Rc::clone(&sps),
    };
    let pps = Rc::new(pps);

    let vps = Vps {
        video_parameter_set_id: 0,
        base_layer_internal_flag: true,
        base_layer_available_flag: true,
        max_layers_minus1: 0,
        max_sub_layers_minus1: 0,
        temporal_id_nesting_flag: true,
        profile_tier_level: ptl,
        sub_layer_ordering_info_present_flag: true,
        max_dec_pic_buffering_minus1: [
            max_dec_pic_buffering_minus1[0] as u32,
            0,
            0,
            0,
            0,
            0,
            0,
        ],
        max_num_reorder_pics: [0; 7],
        max_latency_increase_plus1: [0; 7],
        max_layer_id: 0,
        num_layer_sets_minus1: 0,
        timing_info_present_flag: false,
        extension_flag: false,
        ..Default::default()
    };

    (Rc::new(vps), sps, pps)
}

/// The `pic_init_qp` for the stream: the requested constant-QP value clamped into
/// the tunings' quality bounds, else the midpoint of the bounds.
fn pick_init_qp(tunings: &Tunings) -> u8 {
    let min_qp = tunings.min_quality.max(MIN_QP as u32);
    // Raise `max_qp` to at least `min_qp` before clamping: a fork caller may set
    // `min_quality > 51`, which would leave `min_qp > max_qp` and panic
    // `u32::clamp`, so keep the helper total for all callers.
    let max_qp = tunings.max_quality.min(MAX_QP as u32).max(min_qp);
    if let RateControl::ConstantQuality(init_qp) = tunings.rate_control {
        init_qp.clamp(min_qp, max_qp) as u8
    } else {
        ((min_qp + max_qp) / 2) as u8
    }
}

/// The trivial LowDelay short-term RPS: one negative reference at `delta_poc ==
/// curr_poc's predecessor`, used by the current picture.
pub(crate) fn low_delay_rps(curr_poc: i32, ref_poc: i32) -> ShortTermRefPicSet {
    let mut rps = ShortTermRefPicSet {
        inter_ref_pic_set_prediction_flag: false,
        num_negative_pics: 1,
        num_positive_pics: 0,
        num_delta_pocs: 1,
        ..Default::default()
    };
    rps.delta_poc_s0[0] = ref_poc - curr_poc;
    rps.used_by_curr_pic_s0[0] = true;
    rps
}

pub(crate) struct LowDelayH265Delegate {
    /// Current sequence VPS.
    vps: Option<Rc<Vps>>,
    /// Current sequence SPS.
    sps: Option<Rc<Sps>>,
    /// Current sequence PPS.
    pps: Option<Rc<Pps>>,

    /// The predictor `counter` at which the current coded video sequence (CVS)
    /// began — i.e. the counter of its IDR. Picture order counts are taken
    /// relative to it (`poc = counter - poc_base`) so that every IDR (natural or
    /// `force_keyframe`-driven) is `poc == 0`, matching the decoder's POC reset at
    /// an IDR. Without this, a forced mid-GOP IDR (which resets the decoder POC to
    /// 0) would keep a non-zero encoder POC and the following P frames' RPS would
    /// reference a POC that no longer resolves.
    poc_base: usize,

    /// Encoder config.
    config: EncoderConfig,
}

pub(crate) type LowDelayH265<Picture, Reference> = LowDelay<
    Picture,
    DpbEntry<Reference>,
    LowDelayH265Delegate,
    BackendRequest<Picture, Reference>,
>;

impl<Picture, Reference> LowDelayH265<Picture, Reference> {
    pub(super) fn new(config: EncoderConfig, limit: u16) -> Self {
        Self {
            queue: Default::default(),
            references: Default::default(),
            counter: 0,
            limit,
            tunings: config.initial_tunings.clone(),
            delegate: LowDelayH265Delegate {
                config,
                vps: None,
                sps: None,
                pps: None,
                poc_base: 0,
            },
            tunings_queue: Default::default(),
            _phantom: Default::default(),
        }
    }

    fn new_sequence(&mut self) {
        trace!("beginning new HEVC sequence");
        let (vps, sps, pps) =
            build_parameter_sets(&self.delegate.config, &self.tunings, self.limit);
        self.delegate.vps = Some(vps);
        self.delegate.sps = Some(sps);
        self.delegate.pps = Some(pps);
    }

    /// The number of CTUs in the (single) slice — `PicWidthInCtbsY *
    /// PicHeightInCtbsY`.
    fn num_ctu_in_slice(sps: &Sps) -> u32 {
        sps.pic_width_in_ctbs_y * sps.pic_height_in_ctbs_y
    }
}

impl<Picture, Reference>
    LowDelayDelegate<Picture, DpbEntry<Reference>, BackendRequest<Picture, Reference>>
    for LowDelayH265<Picture, Reference>
{
    fn request_keyframe(
        &mut self,
        input: Picture,
        input_meta: FrameMetadata,
        idr: bool,
    ) -> EncodeResult<BackendRequest<Picture, Reference>> {
        if idr {
            // Natural GOP-boundary IDR (`self.counter == 0`): (re)build the
            // parameter sets for the new sequence. A `force_keyframe`-driven
            // keyframe (`idr == false`, counter != 0) reuses the existing sets.
            self.new_sequence();
        }

        let vps = self.delegate.vps.clone().ok_or(EncodeError::InvalidInternalState)?;
        let sps = self.delegate.sps.clone().ok_or(EncodeError::InvalidInternalState)?;
        let pps = self.delegate.pps.clone().ok_or(EncodeError::InvalidInternalState)?;

        // Every keyframe request — the natural GOP IDR *and* a `force_keyframe`
        // one (which the generic `LowDelay` passes with `idr == false` because the
        // counter is non-zero) — is coded as a TRUE IDR that starts a new coded
        // video sequence. Rebase the POC so the IDR is `poc == 0` (an IDR resets
        // the decoder's POC to 0; the encoder must match) and the following P
        // frames count from it via `request_interframe`.
        self.delegate.poc_base = self.counter;
        let poc = 0i32;
        let dpb_meta = DpbEntryMeta { poc, nalu_type: NaluType::IdrWRadl };

        let header = SliceHeader {
            first_slice_segment_in_pic_flag: true,
            no_output_of_prior_pics_flag: false,
            pic_parameter_set_id: 0,
            type_: SliceType::I,
            pic_output_flag: true,
            // Not coded for an IDR slice (the synthesizer omits pic_order_cnt_lsb
            // for IDR_W_RADL / IDR_N_LP; POC is inferred 0), but carried for the VA
            // picture parameter buffer.
            pic_order_cnt_lsb: 0,
            loop_filter_across_slices_enabled_flag: true,
            qp_delta: 0,
            five_minus_max_num_merge_cand: 0,
            ..Default::default()
        };

        let num_ctu_in_slice = Self::num_ctu_in_slice(&sps);

        let request = BackendRequest {
            vps,
            sps,
            pps,
            header,
            nalu_type: NaluType::IdrWRadl,
            input,
            input_meta,
            dpb_meta,
            // IDR: no references.
            ref_list_0: vec![],
            intra_period: self.limit as u32,
            ip_period: 1,
            num_ctu_in_slice,
            // A keyframe is always a true IDR (new CVS): VPS + SPS are re-emitted,
            // the coding type is intra, and the POC is reset.
            is_idr: true,
            tunings: self.tunings.clone(),
            coded_output: vec![],
        };

        Ok(request)
    }

    fn request_interframe(
        &mut self,
        input: Picture,
        input_meta: FrameMetadata,
    ) -> EncodeResult<BackendRequest<Picture, Reference>> {
        let vps = self.delegate.vps.clone().ok_or(EncodeError::InvalidInternalState)?;
        let sps = self.delegate.sps.clone().ok_or(EncodeError::InvalidInternalState)?;
        let pps = self.delegate.pps.clone().ok_or(EncodeError::InvalidInternalState)?;

        // LowDelay P: reference the single most-recently reconstructed frame.
        let reference =
            self.references.back().cloned().ok_or(EncodeError::InvalidInternalState)?;
        // POC relative to the current CVS's IDR (see `poc_base`). Within a CVS the
        // counter is always >= poc_base (a wrap to counter 0 takes the keyframe
        // path), so this cannot underflow.
        let poc = (self.counter - self.delegate.poc_base) as i32;
        let ref_poc = reference.meta.poc;
        let max_poc_lsb = 1u32 << (sps.log2_max_pic_order_cnt_lsb_minus4 as u32 + 4);

        let dpb_meta = DpbEntryMeta { poc, nalu_type: NaluType::TrailR };

        let mut header = SliceHeader {
            first_slice_segment_in_pic_flag: true,
            pic_parameter_set_id: 0,
            type_: SliceType::P,
            pic_output_flag: true,
            pic_order_cnt_lsb: (poc as u32 % max_poc_lsb) as u16,
            // In-header trivial RPS (short_term_ref_pic_set_sps_flag == 0).
            short_term_ref_pic_set_sps_flag: false,
            short_term_ref_pic_set: low_delay_rps(poc, ref_poc),
            num_ref_idx_active_override_flag: false,
            num_ref_idx_l0_active_minus1: 0,
            num_ref_idx_l1_active_minus1: 0,
            five_minus_max_num_merge_cand: 0,
            loop_filter_across_slices_enabled_flag: true,
            qp_delta: 0,
            ..Default::default()
        };
        header.curr_rps_idx = sps.num_short_term_ref_pic_sets;

        let num_ctu_in_slice = Self::num_ctu_in_slice(&sps);

        let request = BackendRequest {
            vps,
            sps,
            pps,
            header,
            nalu_type: NaluType::TrailR,
            input,
            input_meta,
            dpb_meta,
            ref_list_0: vec![reference],
            intra_period: self.limit as u32,
            ip_period: 1,
            num_ctu_in_slice,
            is_idr: false,
            tunings: self.tunings.clone(),
            coded_output: vec![],
        };

        // A single reference per P frame: the next frame references only this one
        // once it has been reconstructed.
        self.references.clear();

        Ok(request)
    }

    fn try_tunings(&self, _tunings: &Tunings) -> EncodeResult<()> {
        Ok(())
    }

    fn apply_tunings(&mut self, _tunings: &Tunings) -> EncodeResult<()> {
        self.new_sequence();
        Ok(())
    }
}
