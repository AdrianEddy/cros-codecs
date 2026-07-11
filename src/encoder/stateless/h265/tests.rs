// Copyright 2024 The ChromiumOS Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

//! Host (GPU-independent) unit tests for the HEVC encoder: the packed-header
//! attribute decision (iHD-packed vs Mesa-none), and the predictor's VPS / SPS /
//! PPS / slice-header field selection round-tripped through the M7b synthesizer.

use std::io::Cursor;
use std::rc::Rc;

use crate::codec::h264::nalu::Nalu;
use crate::codec::h265::parser::Level;
use crate::codec::h265::parser::NaluHeader;
use crate::codec::h265::parser::NaluType;
use crate::codec::h265::parser::Parser;
use crate::codec::h265::parser::Pps;
use crate::codec::h265::parser::Profile;
use crate::codec::h265::parser::SliceHeader;
use crate::codec::h265::parser::SliceType;
use crate::codec::h265::parser::Sps;
use crate::codec::h265::parser::Vps;
use crate::codec::h265::synthesizer::Synthesizer;
use crate::encoder::h265::EncoderConfig;
use crate::encoder::stateless::h265::predictor::build_parameter_sets;
use crate::encoder::stateless::h265::predictor::low_delay_rps;
use crate::encoder::stateless::h265::predictor::LowDelayH265;
use crate::encoder::stateless::h265::vaapi::coding_type_for_slice;
use crate::encoder::stateless::h265::vaapi::decide_packed_headers;
use crate::encoder::stateless::h265::DpbEntry;
use crate::encoder::stateless::h265::DpbEntryMeta;
use crate::encoder::stateless::predictor::LowDelayDelegate;
use crate::encoder::stateless::FrameMetadata;
use crate::encoder::PredictionStructure;
use crate::encoder::RateControl;
use crate::encoder::Tunings;
use crate::Fourcc;
use crate::FrameLayout;
use crate::Resolution;

fn config(width: u32, height: u32, limit: u16) -> EncoderConfig {
    EncoderConfig {
        resolution: Resolution { width, height },
        profile: Profile::Main,
        level: Level::L4,
        pred_structure: PredictionStructure::LowDelay { limit },
        initial_tunings: Tunings::default(),
    }
}

fn frame_meta(timestamp: u64, force_keyframe: bool) -> FrameMetadata {
    FrameMetadata {
        timestamp,
        layout: FrameLayout {
            format: (Fourcc::from(b"NV12"), 0),
            size: Resolution { width: 1920, height: 1080 },
            planes: vec![],
        },
        force_keyframe,
    }
}

/// A stand-in reconstructed reference at `poc` for driving `request_interframe`
/// in the host tests (the predictor only reads its `meta.poc`).
fn recon_ref(poc: i32) -> Rc<DpbEntry<()>> {
    Rc::new(DpbEntry { recon_pic: (), meta: DpbEntryMeta { poc, nalu_type: NaluType::TrailR } })
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

fn next_nalu(buf: &[u8], expect: NaluType) -> Nalu<'_, NaluHeader> {
    let mut cursor = Cursor::new(buf);
    let nalu = Nalu::<NaluHeader>::next(&mut cursor).unwrap();
    assert_eq!(nalu.header.type_, expect);
    assert_eq!(nalu.header.nuh_layer_id, 0);
    assert_eq!(nalu.header.nuh_temporal_id_plus1, 1);
    nalu
}

// ─────────────────────────────────────────────────────────────────────
// Packed-header decision (§3) — iHD packs all three, Mesa packs none.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn packed_headers_ihd_all_three() {
    let all = libva::VA_ENC_PACKED_HEADER_SEQUENCE
        | libva::VA_ENC_PACKED_HEADER_PICTURE
        | libva::VA_ENC_PACKED_HEADER_SLICE;
    // iHD advertises all three (possibly with extra bits) → app packs all three.
    assert_eq!(decide_packed_headers(all), all);
    assert_eq!(decide_packed_headers(all | libva::VA_ENC_PACKED_HEADER_MISC), all);
}

#[test]
fn packed_headers_mesa_none() {
    // Mesa advertises the attribute as unsupported → the driver self-generates.
    assert_eq!(decide_packed_headers(libva::VA_ATTRIB_NOT_SUPPORTED), libva::VA_ENC_PACKED_HEADER_NONE);
    // A bare NONE value likewise.
    assert_eq!(decide_packed_headers(libva::VA_ENC_PACKED_HEADER_NONE), libva::VA_ENC_PACKED_HEADER_NONE);
}

#[test]
fn packed_headers_partial_is_none() {
    // The all-three-or-none rule: SEQUENCE|PICTURE without SLICE → pack nothing.
    let partial = libva::VA_ENC_PACKED_HEADER_SEQUENCE | libva::VA_ENC_PACKED_HEADER_PICTURE;
    assert_eq!(decide_packed_headers(partial), libva::VA_ENC_PACKED_HEADER_NONE);
    assert_eq!(decide_packed_headers(libva::VA_ENC_PACKED_HEADER_SLICE), libva::VA_ENC_PACKED_HEADER_NONE);
}

// ─────────────────────────────────────────────────────────────────────
// Predictor VPS / SPS / PPS field selection → synthesizer round-trip.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn predictor_vps_round_trips() {
    let (vps, _sps, _pps) = build_parameter_sets(&config(1920, 1080, 60), &Tunings::default(), 60);

    let buf1 = synth_vps(&vps);
    let nalu = next_nalu(&buf1, NaluType::VpsNut);
    let mut parser = Parser::default();
    let parsed = (*parser.parse_vps(&nalu).unwrap()).clone();

    assert_eq!(parsed.video_parameter_set_id, 0);
    assert_eq!(parsed.max_sub_layers_minus1, 0);
    assert_eq!(parsed.profile_tier_level.general_profile_idc, 1);
    assert_eq!(parsed.profile_tier_level.general_level_idc, Level::L4);
    assert_eq!(parsed.max_dec_pic_buffering_minus1[0], 1);
    assert!(!parsed.timing_info_present_flag);

    assert_eq!(buf1, synth_vps(&parsed), "VPS byte-idempotence");
}

#[test]
fn predictor_sps_round_trips_main_420_8bit() {
    let (_vps, sps, _pps) = build_parameter_sets(&config(1920, 1080, 60), &Tunings::default(), 60);

    let buf1 = synth_sps(&sps);
    let nalu = next_nalu(&buf1, NaluType::SpsNut);
    let mut parser = Parser::default();
    let parsed = (*parser.parse_sps(&nalu).unwrap()).clone();

    assert_eq!(parsed.profile_tier_level.general_profile_idc, 1, "Main");
    assert_eq!(parsed.profile_tier_level.general_level_idc, Level::L4);
    assert_eq!(parsed.chroma_format_idc, 1, "4:2:0");
    assert_eq!(parsed.bit_depth_luma_minus8, 0, "8-bit");
    assert_eq!(parsed.bit_depth_chroma_minus8, 0);
    assert_eq!(parsed.pic_width_in_luma_samples, 1920);
    assert_eq!(parsed.pic_height_in_luma_samples, 1080);
    // 1920x1080 are both multiples of MinCbSizeY (8) → no conformance window.
    assert!(!parsed.conformance_window_flag);
    assert_eq!(parsed.num_short_term_ref_pic_sets, 0, "RPS lives in the slice header");
    assert!(!parsed.sample_adaptive_offset_enabled_flag, "SAO off");
    assert!(!parsed.temporal_mvp_enabled_flag, "temporal MVP off");
    assert!(parsed.amp_enabled_flag, "AMP on");
    assert_eq!(parsed.max_dec_pic_buffering_minus1[0], 1);
    assert_eq!(parsed.max_num_reorder_pics[0], 0, "LowDelay — no reordering");
    // MinCbSizeY 8, CtbSizeY 32.
    assert_eq!(parsed.log2_min_luma_coding_block_size_minus3, 0);
    assert_eq!(parsed.log2_diff_max_min_luma_coding_block_size, 2);

    assert_eq!(buf1, synth_sps(&parsed), "SPS byte-idempotence");
}

#[test]
fn predictor_sps_conformance_window_for_non_8_multiple() {
    // 1922x1080: width is even but not a multiple of MinCbSizeY (8). The coded
    // width pads up to 1928; the conformance window crops the 6 luma columns
    // (3 chroma samples at SubWidthC = 2) back to the displayed 1922.
    let (_vps, sps, _pps) = build_parameter_sets(&config(1922, 1080, 60), &Tunings::default(), 60);

    let buf1 = synth_sps(&sps);
    let nalu = next_nalu(&buf1, NaluType::SpsNut);
    let mut parser = Parser::default();
    let parsed = (*parser.parse_sps(&nalu).unwrap()).clone();

    assert_eq!(parsed.pic_width_in_luma_samples, 1928, "padded up to MinCbSizeY");
    assert_eq!(parsed.pic_height_in_luma_samples, 1080);
    assert!(parsed.conformance_window_flag);
    assert_eq!(parsed.conf_win_right_offset, 3, "(1928 - 1922) / SubWidthC");
    assert_eq!(parsed.conf_win_bottom_offset, 0);

    assert_eq!(buf1, synth_sps(&parsed), "conformance-window SPS byte-idempotence");
}

#[test]
fn predictor_pps_round_trips() {
    // ConstantQuality QP 27 → pic_init_qp 27 → init_qp_minus26 == 1.
    let tunings = Tunings { rate_control: RateControl::ConstantQuality(27), ..Default::default() };
    let (_vps, sps, pps) = build_parameter_sets(&config(1920, 1080, 60), &tunings, 60);

    // Register the SPS first (the parser resolves the PPS's SPS reference).
    let sps_buf = synth_sps(&sps);
    let pps_buf = synth_pps(&pps);
    let mut parser = Parser::default();
    parser.parse_sps(&next_nalu(&sps_buf, NaluType::SpsNut)).unwrap();
    let parsed = (*parser.parse_pps(&next_nalu(&pps_buf, NaluType::PpsNut)).unwrap()).clone();

    assert_eq!(parsed.pic_parameter_set_id, 0);
    assert_eq!(parsed.seq_parameter_set_id, 0);
    assert!(parsed.cu_qp_delta_enabled_flag, "CU-QP-delta on for CBR RC");
    assert_eq!(parsed.init_qp_minus26, 1, "pic_init_qp 27 - 26");
    assert!(!parsed.weighted_pred_flag);
    assert!(!parsed.tiles_enabled_flag);
    assert!(!parsed.transform_skip_enabled_flag);

    assert_eq!(pps_buf, synth_pps(&parsed), "PPS byte-idempotence");
}

// ─────────────────────────────────────────────────────────────────────
// P slice header with the trivial LowDelay in-header RPS.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn predictor_p_slice_header_round_trips_trivial_rps() {
    let (_vps, sps, pps) = build_parameter_sets(&config(1920, 1080, 60), &Tunings::default(), 60);

    // Register SPS + PPS so the parser can decode the (SPS/PPS-gated) slice header.
    let sps_buf = synth_sps(&sps);
    let pps_buf = synth_pps(&pps);
    let mut parser = Parser::default();
    parser.parse_sps(&next_nalu(&sps_buf, NaluType::SpsNut)).unwrap();
    parser.parse_pps(&next_nalu(&pps_buf, NaluType::PpsNut)).unwrap();
    let sps_rc: Rc<Sps> = parser.get_sps(0).unwrap().clone();
    let pps_rc: Rc<Pps> = parser.get_pps(0).unwrap().clone();

    // A P frame at POC 5 referencing POC 4: trivial RPS delta_poc_s0 == -1.
    let hdr = SliceHeader {
        first_slice_segment_in_pic_flag: true,
        pic_parameter_set_id: 0,
        type_: SliceType::P,
        pic_output_flag: true,
        pic_order_cnt_lsb: 5,
        short_term_ref_pic_set_sps_flag: false,
        short_term_ref_pic_set: low_delay_rps(5, 4),
        num_ref_idx_active_override_flag: false,
        five_minus_max_num_merge_cand: 0,
        loop_filter_across_slices_enabled_flag: true,
        qp_delta: 0,
        ..Default::default()
    };

    let mut buf1 = Vec::new();
    Synthesizer::<'_, SliceHeader, _>::synthesize(NaluType::TrailR, &hdr, &sps_rc, &pps_rc, &mut buf1, true).unwrap();

    let mut cursor = Cursor::new(&buf1[..]);
    let nalu = Nalu::<NaluHeader>::next(&mut cursor).unwrap();
    assert_eq!(nalu.header.type_, NaluType::TrailR);
    let parsed = parser.parse_slice_header(nalu).unwrap().header;

    assert_eq!(parsed.type_, SliceType::P);
    assert_eq!(parsed.pic_order_cnt_lsb, 5);
    assert!(!parsed.short_term_ref_pic_set_sps_flag);
    assert_eq!(parsed.short_term_ref_pic_set.num_negative_pics, 1);
    assert_eq!(parsed.short_term_ref_pic_set.num_positive_pics, 0);
    assert_eq!(parsed.short_term_ref_pic_set.delta_poc_s0[0], -1);
    assert!(parsed.short_term_ref_pic_set.used_by_curr_pic_s0[0]);
    // NumPicTotalCurr == 1 (one used reference).
    assert_eq!(parsed.num_pic_total_curr, 1);

    let mut buf2 = Vec::new();
    Synthesizer::<'_, SliceHeader, _>::synthesize(NaluType::TrailR, &parsed, &sps_rc, &pps_rc, &mut buf2, true).unwrap();
    assert_eq!(buf1, buf2, "P slice byte-idempotence");
}

// ─────────────────────────────────────────────────────────────────────
// Forced mid-GOP keyframe (Finding 1): a true IDR with the POC reset, and
// mutually consistent NAL type / slice type / coding_type / is_idr / ref list.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn forced_mid_gop_keyframe_is_a_consistent_true_idr() {
    const LIMIT: u16 = 30;
    let mut pred: LowDelayH265<(), ()> = LowDelayH265::new(config(1920, 1080, LIMIT), LIMIT);

    // The generic `LowDelay::next_request` calls `request_keyframe(_, _, counter
    // == 0)` and, after each request, advances `counter = (counter + 1) % limit`.
    // We drive the delegate directly and mirror that counter bookkeeping.

    // Frame 0 — the natural GOP IDR (counter 0 → idr = true).
    let idr = pred.request_keyframe((), frame_meta(0, false), true).unwrap();
    assert!(idr.is_idr);
    assert_eq!(idr.nalu_type, NaluType::IdrWRadl);
    assert_eq!(idr.header.type_, SliceType::I);
    assert_eq!(idr.dpb_meta.poc, 0);
    assert!(idr.ref_list_0.is_empty(), "IDR has no references");
    assert_eq!(coding_type_for_slice(idr.header.type_), 1, "I slice → CODING_TYPE_I");
    pred.counter = 1;

    // Frames 1..5 — ordinary P frames; POC counts up 1, 2, 3, 4.
    for c in 1..5u32 {
        pred.references.clear();
        pred.references.push_back(recon_ref((c - 1) as i32));
        let p = pred.request_interframe((), frame_meta(c as u64, false)).unwrap();
        assert!(!p.is_idr);
        assert_eq!(p.nalu_type, NaluType::TrailR);
        assert_eq!(p.header.type_, SliceType::P);
        assert_eq!(p.dpb_meta.poc, c as i32, "POC counts up within the CVS");
        assert_eq!(p.header.short_term_ref_pic_set.delta_poc_s0[0], -1);
        assert_eq!(coding_type_for_slice(p.header.type_), 2, "P slice → CODING_TYPE_P");
        pred.counter = (c + 1) as usize;
    }

    // Frame 5 — a `force_keyframe` mid-GOP. The generic layer requests it with
    // `idr == false` (counter != 0). It MUST still be coded as a true IDR whose
    // POC is reset to 0 (an IDR resets the decoder's POC to 0).
    pred.references.clear();
    let forced = pred.request_keyframe((), frame_meta(5, true), false).unwrap();
    assert!(forced.is_idr, "a forced mid-GOP keyframe is a true IDR, not a P");
    assert_eq!(forced.nalu_type, NaluType::IdrWRadl, "IDR NAL 19 → seek point + stss");
    assert_eq!(forced.header.type_, SliceType::I, "intra slice");
    assert_eq!(forced.dpb_meta.poc, 0, "POC reset to 0 for the new CVS");
    assert_eq!(forced.header.pic_order_cnt_lsb, 0);
    assert!(forced.ref_list_0.is_empty(), "IDR has an empty ref list");
    // The exact contradiction from Finding 1 (coding_type P for an IDR/I frame)
    // cannot recur: coding_type tracks the slice type.
    assert_eq!(coding_type_for_slice(forced.header.type_), 1, "I slice → CODING_TYPE_I");
    pred.counter = 6;

    // Frame 6 — the P frame after the forced IDR must count from the reset base
    // (POC 1) and reference the IDR (delta_poc == -1 → POC 0), which still
    // resolves in the decoder's DPB.
    pred.references.clear();
    pred.references.push_back(recon_ref(0));
    let after = pred.request_interframe((), frame_meta(6, false)).unwrap();
    assert!(!after.is_idr);
    assert_eq!(after.header.type_, SliceType::P);
    assert_eq!(after.dpb_meta.poc, 1, "POC counts from the reset IDR base, not the raw counter");
    assert_eq!(after.header.pic_order_cnt_lsb, 1);
    assert_eq!(after.header.short_term_ref_pic_set.delta_poc_s0[0], -1);
    assert_eq!(coding_type_for_slice(after.header.type_), 2, "P slice → CODING_TYPE_P");
}
