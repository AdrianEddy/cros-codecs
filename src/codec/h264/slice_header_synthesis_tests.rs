// Copyright 2026 The ChromiumOS Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

//! `slice_header()` synthesis, read back by this crate's own parser: every syntax element the
//! encoder writes decodes to the value it was given, and the length the synthesizer reports is
//! where the parser finds the header ends.

use std::io::Cursor;
use std::rc::Rc;

use super::Synthesizer;
use super::SynthesizerError;
use super::SynthesizerResult;
use crate::codec::h264::parser::Level;
use crate::codec::h264::parser::MaxLongTermFrameIdx;
use crate::codec::h264::parser::Nalu;
use crate::codec::h264::parser::NaluType;
use crate::codec::h264::parser::Parser;
use crate::codec::h264::parser::Pps;
use crate::codec::h264::parser::PpsBuilder;
use crate::codec::h264::parser::Profile;
use crate::codec::h264::parser::RefPicListModification;
use crate::codec::h264::parser::RefPicMarkingInner;
use crate::codec::h264::parser::SliceHeader;
use crate::codec::h264::parser::SliceHeaderBuilder;
use crate::codec::h264::parser::SliceType;
use crate::codec::h264::parser::Sps;
use crate::codec::h264::parser::SpsBuilder;

/// The 4-byte start code the synthesizer opens every NAL unit with.
const START_CODE_BITS: usize = 32;

/// 1280x720 Main, frame coding, POC type 0, with `MaxFrameNum` 16 and `MaxPicOrderCntLsb` 32.
fn sps() -> Rc<Sps> {
    sps_with(|_| {})
}

/// [`sps`], changed by `edit`.
fn sps_with(edit: impl FnOnce(&mut Sps)) -> Rc<Sps> {
    let built = SpsBuilder::new()
        .seq_parameter_set_id(0)
        .profile_idc(Profile::Main)
        .level_idc(Level::L4)
        .resolution(1280, 720)
        .chroma_format_idc(1)
        .frame_mbs_only_flag(true)
        .direct_8x8_inference_flag(true)
        .max_num_ref_frames(4)
        .max_frame_num(16)
        .pic_order_cnt_type(0)
        .max_pic_order_cnt_lsb(32)
        .bit_depth_luma(8)
        .bit_depth_chroma(8)
        .build();
    let mut sps = Rc::try_unwrap(built).expect("the builder's SPS is not shared");
    edit(&mut sps);
    Rc::new(sps)
}

/// A PPS on `sps`, changed by `edit`.
fn pps(sps: &Rc<Sps>, edit: impl FnOnce(&mut Pps)) -> Rc<Pps> {
    let built = PpsBuilder::new(Rc::clone(sps))
        .pic_parameter_set_id(0)
        .pic_init_qp(26)
        .deblocking_filter_control_present_flag(true)
        .num_ref_idx_l0_default_active(1)
        .build();
    let mut pps = Rc::try_unwrap(built).expect("the builder's PPS is not shared");
    edit(&mut pps);
    Rc::new(pps)
}

/// A parser holding `sps` and `pps`, read from their own synthesized NAL units.
fn parser_for(sps: &Sps, pps: &Pps) -> Parser {
    let mut bytes = Vec::new();
    Synthesizer::<'_, Sps, _>::synthesize(3, sps, &mut bytes, true).unwrap();
    Synthesizer::<'_, Pps, _>::synthesize(3, pps, &mut bytes, true).unwrap();
    let mut parser = Parser::default();
    let mut cursor = Cursor::new(&bytes[..]);
    let nalu = Nalu::next(&mut cursor).unwrap();
    parser.parse_sps(&nalu).unwrap();
    let nalu = Nalu::next(&mut cursor).unwrap();
    parser.parse_pps(&nalu).unwrap();
    parser
}

/// Synthesizes `header`, checks the reported length, and reads it back.
fn round_trip(
    nal_ref_idc: u8,
    nalu_type: NaluType,
    header: &SliceHeader,
    sps: &Sps,
    pps: &Pps,
) -> SliceHeader {
    let mut packed = Vec::new();
    let padding = Synthesizer::<'_, SliceHeader, _>::synthesize(
        nal_ref_idc,
        nalu_type,
        header,
        sps,
        pps,
        &mut packed,
        true,
    )
    .unwrap();
    assert!(padding < 8, "a byte is completed with at most 7 bits, got {padding}");
    let bits = packed.len() * 8 - usize::from(padding);

    // The slice data continues the header; give it a first bit, as a driver's would be, so the
    // parser reads the header's last byte whole whatever the padding holds.
    if padding == 0 {
        packed.push(0x80);
    } else {
        *packed.last_mut().unwrap() |= 1 << (padding - 1);
    }

    let parser = parser_for(sps, pps);
    let mut cursor = Cursor::new(&packed[..]);
    let nalu = Nalu::next(&mut cursor).unwrap();
    assert_eq!(nalu.header.type_, nalu_type, "nal_unit_type");
    assert_eq!(nalu.header.ref_idc, nal_ref_idc, "nal_ref_idc");
    let slice = parser.parse_slice_header(nalu).unwrap();
    // The packed length counts the emulation prevention bytes it holds
    // (`has_emulation_bytes`); the parser's counts the header's own bits.
    assert_eq!(
        slice.header.header_bit_size,
        bits - START_CODE_BITS - 8 * slice.header.n_emulation_prevention_bytes,
        "the packed header's length is where the parser finds the slice data begins"
    );
    slice.header
}

/// The modifications of one list, without the closing `modification_of_pic_nums_idc` 3 the
/// parser keeps.
fn modifications(ops: &[RefPicListModification]) -> Vec<RefPicListModification> {
    ops.iter().filter(|op| op.modification_of_pic_nums_idc != 3).cloned().collect()
}

/// The syntax elements a slice header carries, compared one by one (the parser also derives
/// sizes and inferred values the synthesizer does not take).
fn assert_same_syntax(written: &SliceHeader, read: &SliceHeader, max_frame_num: u32, max_lsb: u32) {
    assert_eq!(read.first_mb_in_slice, written.first_mb_in_slice, "first_mb_in_slice");
    assert_eq!(read.slice_type, written.slice_type, "slice_type");
    assert_eq!(read.pic_parameter_set_id, written.pic_parameter_set_id, "pic_parameter_set_id");
    assert_eq!(
        u32::from(read.frame_num),
        u32::from(written.frame_num) % max_frame_num,
        "frame_num"
    );
    assert_eq!(read.field_pic_flag, written.field_pic_flag, "field_pic_flag");
    assert_eq!(read.bottom_field_flag, written.bottom_field_flag, "bottom_field_flag");
    assert_eq!(read.idr_pic_id, written.idr_pic_id, "idr_pic_id");
    assert_eq!(
        u32::from(read.pic_order_cnt_lsb),
        u32::from(written.pic_order_cnt_lsb) % max_lsb,
        "POC lsb"
    );
    assert_eq!(
        read.delta_pic_order_cnt_bottom, written.delta_pic_order_cnt_bottom,
        "delta_pic_order_cnt_bottom"
    );
    assert_eq!(read.delta_pic_order_cnt, written.delta_pic_order_cnt, "delta_pic_order_cnt");
    assert_eq!(read.redundant_pic_cnt, written.redundant_pic_cnt, "redundant_pic_cnt");
    assert_eq!(
        read.direct_spatial_mv_pred_flag, written.direct_spatial_mv_pred_flag,
        "direct_spatial_mv_pred"
    );
    assert_eq!(
        read.num_ref_idx_active_override_flag, written.num_ref_idx_active_override_flag,
        "num_ref_idx_active_override_flag"
    );
    assert_eq!(
        read.ref_pic_list_modification_flag_l0, written.ref_pic_list_modification_flag_l0,
        "rplm l0 flag"
    );
    assert_eq!(
        modifications(&read.ref_pic_list_modification_l0),
        modifications(&written.ref_pic_list_modification_l0),
        "rplm l0"
    );
    assert_eq!(
        read.ref_pic_list_modification_flag_l1, written.ref_pic_list_modification_flag_l1,
        "rplm l1 flag"
    );
    assert_eq!(
        modifications(&read.ref_pic_list_modification_l1),
        modifications(&written.ref_pic_list_modification_l1),
        "rplm l1"
    );
    assert_eq!(read.dec_ref_pic_marking, written.dec_ref_pic_marking, "dec_ref_pic_marking");
    assert_eq!(read.cabac_init_idc, written.cabac_init_idc, "cabac_init_idc");
    assert_eq!(read.slice_qp_delta, written.slice_qp_delta, "slice_qp_delta");
    assert_eq!(read.sp_for_switch_flag, written.sp_for_switch_flag, "sp_for_switch_flag");
    assert_eq!(read.slice_qs_delta, written.slice_qs_delta, "slice_qs_delta");
    assert_eq!(
        read.disable_deblocking_filter_idc, written.disable_deblocking_filter_idc,
        "deblocking idc"
    );
    assert_eq!(read.slice_alpha_c0_offset_div2, written.slice_alpha_c0_offset_div2, "alpha offset");
    assert_eq!(read.slice_beta_offset_div2, written.slice_beta_offset_div2, "beta offset");
}

/// The encoder's IDR: an I slice opening a sequence, its `idr_pic_id` and the IDR marking.
#[test]
fn an_idr_slice_reads_back() {
    let sps = sps();
    let pps = pps(&sps, |_| {});
    let mut header = SliceHeaderBuilder::new(&pps)
        .slice_type(SliceType::I)
        .first_mb_in_slice(0)
        .idr_pic_id(7)
        .build();
    header.slice_qp_delta = -3;
    header.slice_alpha_c0_offset_div2 = 2;
    header.slice_beta_offset_div2 = -1;
    let read = round_trip(3, NaluType::SliceIdr, &header, &sps, &pps);
    assert_same_syntax(&header, &read, sps.max_frame_num(), 32);
}

/// The encoder's P slice: the previous frame in list 0 by default, sliding-window marking.
/// `frame_num` and the POC lsb past their moduli are written modulo them.
#[test]
fn a_p_slice_reads_back_with_its_counters_wrapped() {
    let sps = sps();
    let pps = pps(&sps, |_| {});
    let mut header = SliceHeaderBuilder::new(&pps)
        .slice_type(SliceType::P)
        .first_mb_in_slice(0)
        .pic_order_cnt_lsb(70)
        .build();
    header.frame_num = 21;
    let read = round_trip(2, NaluType::Slice, &header, &sps, &pps);
    assert_same_syntax(&header, &read, sps.max_frame_num(), 32);
    assert_eq!(read.frame_num, 5, "21 modulo MaxFrameNum 16");
    assert_eq!(read.pic_order_cnt_lsb, 6, "70 modulo MaxPicOrderCntLsb 32");
}

/// A CABAC P slice that overrides the reference count, reorders list 0 and marks with memory
/// management operations: every conditional element the encoder can reach.
#[test]
fn a_cabac_p_slice_with_reordering_and_marking_reads_back() {
    let sps = sps();
    let pps = pps(&sps, |p| p.entropy_coding_mode_flag = true);
    let mut header = SliceHeaderBuilder::new(&pps)
        .slice_type(SliceType::P)
        .first_mb_in_slice(0)
        .pic_order_cnt_lsb(12)
        .num_ref_idx_l0_active(2)
        .build();
    header.frame_num = 6;
    header.cabac_init_idc = 2;
    header.slice_qp_delta = 5;
    header.disable_deblocking_filter_idc = 1;
    header.ref_pic_list_modification_flag_l0 = true;
    header.ref_pic_list_modification_l0 = vec![
        RefPicListModification {
            modification_of_pic_nums_idc: 0,
            abs_diff_pic_num_minus1: 1,
            ..Default::default()
        },
        RefPicListModification {
            modification_of_pic_nums_idc: 2,
            long_term_pic_num: 0,
            ..Default::default()
        },
    ];
    header.dec_ref_pic_marking.adaptive_ref_pic_marking_mode_flag = true;
    header.dec_ref_pic_marking.inner = vec![
        RefPicMarkingInner {
            memory_management_control_operation: 1,
            difference_of_pic_nums_minus1: 3,
            ..Default::default()
        },
        RefPicMarkingInner {
            memory_management_control_operation: 4,
            max_long_term_frame_idx: MaxLongTermFrameIdx::Idx(1),
            ..Default::default()
        },
        RefPicMarkingInner {
            memory_management_control_operation: 3,
            difference_of_pic_nums_minus1: 0,
            long_term_frame_idx: 1,
            ..Default::default()
        },
        RefPicMarkingInner {
            memory_management_control_operation: 6,
            long_term_frame_idx: 0,
            ..Default::default()
        },
    ];
    let read = round_trip(1, NaluType::Slice, &header, &sps, &pps);
    assert_same_syntax(&header, &read, sps.max_frame_num(), 32);
    assert_eq!(read.num_ref_idx_l0_active_minus1, 1, "two references overridden");
}

/// A B slice with explicit weighted bi-prediction, both lists reordered, and no
/// `nal_ref_idc` (no marking).
#[test]
fn a_weighted_b_slice_reads_back() {
    let sps = sps();
    let pps = pps(&sps, |p| {
        p.weighted_bipred_idc = 1;
        p.num_ref_idx_l1_default_active_minus1 = 0;
    });
    let mut header = SliceHeaderBuilder::new(&pps)
        .slice_type(SliceType::B)
        .first_mb_in_slice(0)
        .pic_order_cnt_lsb(5)
        .build();
    header.frame_num = 3;
    header.direct_spatial_mv_pred_flag = true;
    header.ref_pic_list_modification_flag_l1 = true;
    header.ref_pic_list_modification_l1 = vec![RefPicListModification {
        modification_of_pic_nums_idc: 1,
        abs_diff_pic_num_minus1: 0,
        ..Default::default()
    }];
    let pwt = &mut header.pred_weight_table;
    pwt.luma_log2_weight_denom = 5;
    pwt.chroma_log2_weight_denom = 4;
    // List 0: an explicit luma weight, inferred chroma. List 1: inferred luma, explicit chroma.
    pwt.luma_weight_l0[0] = 40;
    pwt.luma_offset_l0[0] = -3;
    pwt.chroma_weight_l0[0] = [16, 16];
    pwt.luma_weight_l1[0] = 32;
    pwt.chroma_weight_l1[0] = [12, 20];
    pwt.chroma_offset_l1[0] = [1, -2];
    let read = round_trip(0, NaluType::Slice, &header, &sps, &pps);
    assert_same_syntax(&header, &read, sps.max_frame_num(), 32);
    assert_eq!(read.pred_weight_table, header.pred_weight_table, "pred_weight_table");
}

/// The emulation prevention the packed header states (`has_emulation_bytes`): a header whose
/// bits hold a start-code pattern gets its `0x03`, and the length still counts every byte.
#[test]
fn a_header_with_a_start_code_pattern_is_escaped() {
    let sps = sps();
    let pps = pps(&sps, |_| {});
    // first_mb_in_slice 2^22 - 1 is ue(v): 22 zero bits, a one, 22 zero bits. Past the header
    // byte that is 0x00, 0x00, then 0x02: a start-code prefix unless escaped.
    let header = SliceHeaderBuilder::new(&pps)
        .slice_type(SliceType::I)
        .first_mb_in_slice((1 << 22) - 1)
        .build();
    let mut packed = Vec::new();
    Synthesizer::<'_, SliceHeader, _>::synthesize(
        3,
        NaluType::SliceIdr,
        &header,
        &sps,
        &pps,
        &mut packed,
        true,
    )
    .unwrap();
    assert!(
        packed[4..].windows(3).any(|w| w == [0x00, 0x00, 0x03]),
        "an emulation prevention byte follows the two zero bytes: {packed:02x?}"
    );
    let read = round_trip(3, NaluType::SliceIdr, &header, &sps, &pps);
    assert_same_syntax(&header, &read, sps.max_frame_num(), 32);
    assert!(read.n_emulation_prevention_bytes > 0, "the parser stripped the escapes it met");
}

/// What the synthesizer refuses rather than write a header no decoder reads as given.
#[test]
fn an_unwritable_header_is_refused() {
    let sps = sps();
    let pps = pps(&sps, |_| {});
    let header = SliceHeaderBuilder::new(&pps).slice_type(SliceType::I).build();
    let synthesize = |ref_idc, nalu_type, pps: &Pps| {
        Synthesizer::<'_, SliceHeader, _>::synthesize(
            ref_idc,
            nalu_type,
            &header,
            &sps,
            pps,
            Vec::new(),
            true,
        )
    };
    assert!(
        matches!(synthesize(0, NaluType::SliceIdr, &pps), Err(SynthesizerError::Invalid(_))),
        "an IDR slice is a reference picture"
    );
    assert!(
        matches!(synthesize(4, NaluType::Slice, &pps), Err(SynthesizerError::Invalid(_))),
        "nal_ref_idc is 2 bits"
    );
    assert!(
        matches!(synthesize(1, NaluType::SliceDpa, &pps), Err(SynthesizerError::Unsupported)),
        "data partitions have no slice header of this form"
    );
    let grouped = self::pps(&sps, |p| p.num_slice_groups_minus1 = 1);
    assert!(
        matches!(synthesize(1, NaluType::Slice, &grouped), Err(SynthesizerError::Unsupported)),
        "slice groups need the group map, which no synthesizer here writes"
    );
}

/// Synthesizes `header` as a non-IDR reference slice.
fn synthesize_slice(header: &SliceHeader, sps: &Sps, pps: &Pps) -> SynthesizerResult<u8> {
    Synthesizer::<'_, SliceHeader, _>::synthesize(
        2,
        NaluType::Slice,
        header,
        sps,
        pps,
        Vec::new(),
        true,
    )
}

fn is_invalid(result: SynthesizerResult<u8>) -> bool {
    matches!(result, Err(SynthesizerError::Invalid(_)))
}

/// A P slice on `pps` overriding the reference count of list 0 to `l0_minus1` + 1.
fn p_slice_with_l0(pps: &Pps, l0_minus1: u8) -> SliceHeader {
    SliceHeaderBuilder::new(pps)
        .slice_type(SliceType::P)
        .num_ref_idx_l0_active_minus1(l0_minus1)
        .build()
}

/// The values past the ranges a slice header is sized, indexed or shifted by are refused, never
/// written or panicked on. Each bound is the parser's too, so what is written reads back.
#[test]
fn a_value_past_its_range_is_refused() {
    let sps = sps();
    let plain = pps(&sps, |_| {});

    // A weighted P slice taking its PPS's default of 33 references: past the 32 entries every
    // list of the weight table holds.
    let weighted_33 = pps(&sps, |p| {
        p.weighted_pred_flag = true;
        p.num_ref_idx_l0_default_active_minus1 = 32;
    });
    let header = SliceHeaderBuilder::new(&weighted_33).slice_type(SliceType::P).build();
    assert!(is_invalid(synthesize_slice(&header, &sps, &weighted_33)), "33 references");

    // A frame slice holds at most 16 references a list (7.4.3).
    assert!(
        is_invalid(synthesize_slice(&p_slice_with_l0(&plain, 16), &sps, &plain)),
        "17 in a frame"
    );
    let read = round_trip(2, NaluType::Slice, &p_slice_with_l0(&plain, 15), &sps, &plain);
    assert_eq!(read.num_ref_idx_l0_active_minus1, 15, "16 in a frame");

    // List 1 is bounded in a B slice, and a P slice codes none.
    let b = SliceHeaderBuilder::new(&plain)
        .slice_type(SliceType::B)
        .num_ref_idx_l1_active_minus1(16)
        .build();
    assert!(is_invalid(synthesize_slice(&b, &sps, &plain)), "17 in list 1 of a B frame slice");
    let p = SliceHeaderBuilder::new(&plain)
        .slice_type(SliceType::P)
        .num_ref_idx_l1_active_minus1(40)
        .build();
    assert!(synthesize_slice(&p, &sps, &plain).is_ok(), "a P slice codes no list 1");

    // A field slice holds up to 32, in a sequence that codes fields; a frame-only sequence has
    // no field slice.
    let fields = sps_with(|s| s.frame_mbs_only_flag = false);
    let field_slice =
        |l0_minus1| SliceHeader { field_pic_flag: true, ..p_slice_with_l0(&plain, l0_minus1) };
    assert!(is_invalid(synthesize_slice(&field_slice(32), &fields, &plain)), "33 in a field");
    let read = round_trip(2, NaluType::Slice, &field_slice(31), &fields, &plain);
    assert!(read.field_pic_flag, "a field slice");
    assert_eq!(read.num_ref_idx_l0_active_minus1, 31, "32 in a field");
    assert!(is_invalid(synthesize_slice(&field_slice(0), &sps, &plain)), "a field of frames");

    // A weight denominator is at most 7 (7.4.3.2); the chroma one is coded, and so bounded,
    // only where chroma is.
    let weighted = pps(&sps, |p| p.weighted_pred_flag = true);
    let mut header = SliceHeaderBuilder::new(&weighted).slice_type(SliceType::P).build();
    header.pred_weight_table.luma_log2_weight_denom = 8;
    assert!(is_invalid(synthesize_slice(&header, &sps, &weighted)), "luma denominator 8");
    header.pred_weight_table.luma_log2_weight_denom = 7;
    header.pred_weight_table.chroma_log2_weight_denom = 8;
    assert!(is_invalid(synthesize_slice(&header, &sps, &weighted)), "chroma denominator 8");
    let monochrome = sps_with(|s| s.chroma_format_idc = 0);
    header.pred_weight_table.chroma_log2_weight_denom = 40;
    assert!(
        synthesize_slice(&header, &monochrome, &weighted).is_ok(),
        "monochrome codes no chroma weight"
    );

    // An explicit weight or offset is in -128..=127 (7.4.3.2); an inferred weight may be 128.
    let mut header = SliceHeaderBuilder::new(&weighted).slice_type(SliceType::P).build();
    header.pred_weight_table.luma_log2_weight_denom = 7;
    header.pred_weight_table.chroma_log2_weight_denom = 7;
    header.pred_weight_table.luma_weight_l0[0] = 128;
    header.pred_weight_table.chroma_weight_l0[0] = [128, 128];
    let read = round_trip(2, NaluType::Slice, &header, &sps, &weighted).pred_weight_table;
    assert_eq!(read.luma_weight_l0[0], 128, "an inferred luma weight of 2^7");
    assert_eq!(read.chroma_weight_l0[0], [128, 128], "inferred chroma weights of 2^7");
    header.pred_weight_table.luma_weight_l0[0] = 127;
    header.pred_weight_table.luma_offset_l0[0] = -128;
    let read = round_trip(2, NaluType::Slice, &header, &sps, &weighted).pred_weight_table;
    assert_eq!((read.luma_weight_l0[0], read.luma_offset_l0[0]), (127, -128), "the luma bounds");
    header.pred_weight_table.luma_weight_l0[0] = 128;
    header.pred_weight_table.luma_offset_l0[0] = 1;
    assert!(is_invalid(synthesize_slice(&header, &sps, &weighted)), "an explicit weight of 128");
    header.pred_weight_table.luma_offset_l0[0] = 0;
    header.pred_weight_table.chroma_weight_l0[0] = [16, -129];
    assert!(is_invalid(synthesize_slice(&header, &sps, &weighted)), "a chroma weight of -129");

    // MaxFrameNum and MaxPicOrderCntLsb are at most 2^16 (7.4.2.1.1); the POC lsb is coded
    // only under POC type 0.
    let p = SliceHeaderBuilder::new(&plain).slice_type(SliceType::P).build();
    let frame_num_17 = sps_with(|s| s.log2_max_frame_num_minus4 = 13);
    assert!(is_invalid(synthesize_slice(&p, &frame_num_17, &plain)), "MaxFrameNum 2^17");
    let lsb_17 = sps_with(|s| s.log2_max_pic_order_cnt_lsb_minus4 = 13);
    assert!(is_invalid(synthesize_slice(&p, &lsb_17, &plain)), "MaxPicOrderCntLsb 2^17");
    let lsb_17_unused = sps_with(|s| {
        s.log2_max_pic_order_cnt_lsb_minus4 = 13;
        s.pic_order_cnt_type = 2;
    });
    assert!(synthesize_slice(&p, &lsb_17_unused, &plain).is_ok(), "POC type 2 codes no lsb");
}
