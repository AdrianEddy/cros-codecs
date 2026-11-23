// Copyright 2024 The ChromiumOS Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

//! Host (GPU-independent) unit tests for the H.264 predictor: the caller's CICP colour
//! description in the SPS VUI (`video_signal_type` + `colour_description`), round-tripped
//! through the synthesizer and the parser; the slices' NAL unit headers and picture counters;
//! and which parameter sets each access unit states, across retunes.

use std::io::Cursor;
use std::rc::Rc;

use crate::codec::h264::parser::Level;
use crate::codec::h264::parser::Nalu;
use crate::codec::h264::parser::NaluType;
use crate::codec::h264::parser::Parser;
use crate::codec::h264::parser::Profile;
use crate::codec::h264::parser::SliceHeader;
use crate::codec::h264::parser::Sps;
use crate::codec::h264::synthesizer::Synthesizer;
use crate::encoder::h264::EncoderConfig;
use crate::encoder::stateless::h264::predictor::LowDelayH264;
use crate::encoder::stateless::h264::BackendRequest;
use crate::encoder::stateless::h264::DpbEntry;
use crate::encoder::stateless::h264::ParameterSets;
use crate::encoder::stateless::predictor::LowDelayDelegate;
use crate::encoder::stateless::FrameMetadata;
use crate::encoder::stateless::Predictor;
use crate::encoder::EncoderColorInfo;
use crate::encoder::PredictionStructure;
use crate::encoder::RateControl;
use crate::encoder::Tunings;
use crate::Fourcc;
use crate::FrameLayout;
use crate::Resolution;

fn config(color: Option<EncoderColorInfo>) -> EncoderConfig {
    EncoderConfig {
        resolution: Resolution { width: 1920, height: 1080 },
        profile: Profile::Main,
        level: Level::L4,
        pred_structure: PredictionStructure::LowDelay { limit: 30 },
        initial_tunings: Tunings::default(),
        color,
        emit_parameter_sets: true,
    }
}

fn frame_meta(timestamp: u64) -> FrameMetadata {
    FrameMetadata {
        timestamp,
        layout: FrameLayout {
            format: (Fourcc::from(b"NV12"), 0),
            size: Resolution { width: 1920, height: 1080 },
            planes: vec![],
        },
        force_keyframe: false,
    }
}

/// Drive the predictor's IDR path and return the SPS it built.
fn keyframe_sps(color: Option<EncoderColorInfo>) -> Rc<Sps> {
    let mut pred: LowDelayH264<(), ()> = LowDelayH264::new(config(color), 30);
    let req = pred.request_keyframe((), frame_meta(0), true).unwrap();
    req.sps
}

fn synth_sps(sps: &Sps) -> Vec<u8> {
    let mut buf = Vec::new();
    Synthesizer::<'_, Sps, _>::synthesize(3, sps, &mut buf, true).unwrap();
    buf
}

fn parse_sps(buf: &[u8]) -> Rc<Sps> {
    let mut cursor = Cursor::new(buf);
    let nalu = Nalu::next(&mut cursor).unwrap();
    assert_eq!(nalu.header.type_, NaluType::Sps);
    let mut parser = Parser::default();
    // `parse_sps` returns `&Rc<Sps>`; clone the `Rc` out (`Sps` is not `Clone`).
    parser.parse_sps(&nalu).unwrap().clone()
}

#[test]
fn no_colour_leaves_video_signal_type_absent() {
    // A VUI exists (aspect ratio and timing) but carries no colour block.
    let sps = keyframe_sps(None);
    let parsed = parse_sps(&synth_sps(&sps));
    assert!(parsed.vui_parameters_present_flag, "VUI present (aspect ratio + timing)");
    assert!(!parsed.vui_parameters.video_signal_type_present_flag, "no colour signalled");
    assert!(!parsed.vui_parameters.colour_description_present_flag);
    assert_eq!(synth_sps(&sps), synth_sps(&parsed), "SPS byte-idempotence");
}

#[test]
fn cicp_colour_survives_round_trip() {
    // BT.709 limited range — the SDR default: CICP 1 / 1 / 1.
    let color = EncoderColorInfo { primaries: 1, transfer: 1, matrix: 1, full_range: false };
    let sps = keyframe_sps(Some(color));
    let parsed = parse_sps(&synth_sps(&sps));

    let vui = &parsed.vui_parameters;
    assert!(vui.video_signal_type_present_flag);
    assert!(!vui.video_full_range_flag, "limited range");
    assert!(vui.colour_description_present_flag);
    assert_eq!(vui.colour_primaries, 1, "BT.709");
    assert_eq!(vui.transfer_characteristics, 1);
    assert_eq!(vui.matrix_coefficients, 1);
    assert_eq!(synth_sps(&sps), synth_sps(&parsed), "SPS byte-idempotence");
}

#[test]
fn emit_parameter_sets_off_yields_empty_coded_output_on_idr() {
    // With `emit_parameter_sets = false` (backend owns the parameter sets),
    // even the IDR request must leave `coded_output` empty — the predictor
    // synthesizes no SPS/PPS, so the backend is free to prepend the
    // driver-authoritative header bytes.
    let mut cfg = config(None);
    cfg.emit_parameter_sets = false;
    let mut pred: LowDelayH264<(), ()> = LowDelayH264::new(cfg, 30);
    let idr = pred.request_keyframe((), frame_meta(0), true).unwrap();
    assert!(idr.is_idr);
    assert!(
        idr.coded_output.is_empty(),
        "IDR coded_output must be empty when emit_parameter_sets is off"
    );
    // A following inter frame is likewise parameter-set-free.
    let inter = pred.request_interframe((), frame_meta(1)).unwrap();
    assert!(
        inter.coded_output.is_empty(),
        "inter coded_output must be empty when emit_parameter_sets is off"
    );
}

#[test]
fn emit_parameter_sets_on_writes_sps_pps_on_idr() {
    // The default (`emit_parameter_sets = true`) still writes SPS+PPS in-band on
    // IDR — the historical behaviour the VAAPI lane relies on.
    let mut pred: LowDelayH264<(), ()> = LowDelayH264::new(config(None), 30);
    let idr = pred.request_keyframe((), frame_meta(0), true).unwrap();
    assert!(
        !idr.coded_output.is_empty(),
        "IDR coded_output carries the in-band SPS/PPS by default"
    );
}

#[test]
fn hdr_full_range_colour_survives_round_trip() {
    // BT.2020 NCL / PQ, full range: CICP 9 / 16 / 9 (an HDR10-style descriptor;
    // colour signalling is independent of the 8-bit coding).
    let color = EncoderColorInfo { primaries: 9, transfer: 16, matrix: 9, full_range: true };
    let sps = keyframe_sps(Some(color));
    let parsed = parse_sps(&synth_sps(&sps));

    let vui = &parsed.vui_parameters;
    assert!(vui.video_full_range_flag, "full range");
    assert_eq!(vui.colour_primaries, 9, "BT.2020");
    assert_eq!(vui.transfer_characteristics, 16, "SMPTE ST 2084 (PQ)");
    assert_eq!(vui.matrix_coefficients, 9, "BT.2020 NCL");
    assert_eq!(synth_sps(&sps), synth_sps(&parsed), "SPS byte-idempotence");
}

/// The predictor at `limit` frames per GOP with no colour.
fn predictor(limit: u16) -> LowDelayH264<(), ()> {
    let mut cfg = config(None);
    cfg.pred_structure = PredictionStructure::LowDelay { limit };
    LowDelayH264::new(cfg, limit)
}

/// The request's slice header as the packed header a VA-API backend sends, read back by a parser
/// that holds the request's own SPS and PPS.
fn read_back_slice(req: &BackendRequest<(), ()>) -> ((u8, NaluType), SliceHeader) {
    use crate::codec::h264::parser::Pps;

    let mut parameter_sets = Vec::new();
    Synthesizer::<'_, Sps, _>::synthesize(3, &req.sps, &mut parameter_sets, true).unwrap();
    Synthesizer::<'_, Pps, _>::synthesize(3, &req.pps, &mut parameter_sets, true).unwrap();
    let mut parser = Parser::default();
    let mut cursor = Cursor::new(&parameter_sets[..]);
    parser.parse_sps(&Nalu::next(&mut cursor).unwrap()).unwrap();
    parser.parse_pps(&Nalu::next(&mut cursor).unwrap()).unwrap();

    let (nal_ref_idc, nalu_type) = req.slice_nal_header();
    let mut packed = Vec::new();
    let padding = Synthesizer::<'_, SliceHeader, _>::synthesize(
        nal_ref_idc,
        nalu_type,
        &req.header,
        &req.sps,
        &req.pps,
        &mut packed,
        true,
    )
    .unwrap();
    // The slice data's first bit, so the header's last byte reads whole.
    if padding == 0 {
        packed.push(0x80);
    } else {
        *packed.last_mut().unwrap() |= 1 << (padding - 1);
    }
    let nalu = Nalu::next(&mut Cursor::new(&packed[..])).unwrap();
    let nal = (nalu.header.ref_idc, nalu.header.type_);
    (nal, parser.parse_slice_header(nalu).unwrap().header)
}

#[test]
fn an_idr_is_a_reference_slice_of_nal_unit_type_5_and_a_p_frame_one_of_type_1() {
    let mut pred = predictor(30);
    let idr = pred.request_keyframe((), frame_meta(0), true).unwrap();
    assert_eq!(idr.slice_nal_header(), (3, NaluType::SliceIdr));
    let (nal, header) = read_back_slice(&idr);
    assert_eq!(nal, (3, NaluType::SliceIdr), "the IDR's NAL unit header");
    assert!(header.slice_type.is_i());

    pred.counter = 1;
    let p = pred.request_interframe((), frame_meta(1)).unwrap();
    assert_eq!(p.slice_nal_header(), (2, NaluType::Slice), "a P frame is a reference picture");
    let (nal, header) = read_back_slice(&p);
    assert_eq!(nal, (2, NaluType::Slice), "the P frame's NAL unit header");
    assert!(header.slice_type.is_p());
}

#[test]
fn slices_count_frame_num_and_poc_through_the_gop() {
    let mut pred = predictor(30);
    let idr = pred.request_keyframe((), frame_meta(0), true).unwrap();
    assert_eq!(
        (idr.header.frame_num, idr.header.pic_order_cnt_lsb),
        (0, 0),
        "an IDR restarts both counters"
    );
    assert_eq!((idr.dpb_meta.frame_num, idr.dpb_meta.poc), (0, 0));
    // MaxFrameNum is 16 and MaxPicOrderCntLsb 32 at a 30-frame GOP: past them the counters wrap.
    assert_eq!((idr.sps.max_frame_num(), idr.sps.max_pic_order_cnt_lsb()), (16, 32));
    for n in 1..30u16 {
        pred.counter = usize::from(n);
        let p = pred.request_interframe((), frame_meta(u64::from(n))).unwrap();
        assert_eq!(p.header.frame_num, n % 16, "frame {n}'s frame_num, modulo MaxFrameNum");
        assert_eq!(p.dpb_meta.frame_num, u32::from(n % 16), "frame {n}'s FrameNum is the same");
        assert_eq!(p.header.pic_order_cnt_lsb, 2 * n % 32, "frame {n}'s POC lsb");
        assert_eq!(p.dpb_meta.poc, 2 * i32::from(n), "frame {n}'s POC, in full");
        let (_, read) = read_back_slice(&p);
        assert_eq!(read.frame_num, p.header.frame_num, "frame {n}'s frame_num read back");
        assert_eq!(read.pic_order_cnt_lsb, p.header.pic_order_cnt_lsb, "frame {n}'s POC read back");
    }
}

#[test]
fn frame_32_of_a_60_frame_gop_wraps_frame_num_and_the_poc_lsb_but_not_the_poc() {
    // Twice 30 fps, the GOP the VA-API and Vulkan Video lanes default to: MaxFrameNum 32 and
    // MaxPicOrderCntLsb 64, the largest powers of two not above 60 and 120.
    let mut pred = predictor(60);
    let idr = pred.request_keyframe((), frame_meta(0), true).unwrap();
    assert_eq!((idr.sps.max_frame_num(), idr.sps.max_pic_order_cnt_lsb()), (32, 64));
    for (n, frame_num, lsb) in [(31u16, 31u16, 62u16), (32, 0, 0), (33, 1, 2), (59, 27, 54)] {
        pred.counter = usize::from(n);
        let p = pred.request_interframe((), frame_meta(u64::from(n))).unwrap();
        assert_eq!(
            (p.header.frame_num, p.header.pic_order_cnt_lsb),
            (frame_num, lsb),
            "frame {n}: the slice header's frame_num and POC lsb"
        );
        // VA-API's `frame_idx` and Vulkan Video's `frame_num` and `FrameNum` take FrameNum; VA's
        // `TopFieldOrderCnt` and Vulkan's `PicOrderCnt` the full POC.
        assert_eq!(p.dpb_meta.frame_num, u32::from(frame_num), "frame {n}'s FrameNum");
        assert_eq!(p.dpb_meta.poc, 2 * i32::from(n), "frame {n}'s POC");
        let (_, read) = read_back_slice(&p);
        assert_eq!(
            (read.frame_num, read.pic_order_cnt_lsb),
            (frame_num, lsb),
            "frame {n} read back"
        );
    }
}

#[test]
fn the_full_poc_of_a_gop_past_32768_frames_is_past_u16() {
    // The longest GOP the predictor takes: MaxFrameNum and MaxPicOrderCntLsb take their
    // largest, 32768 (the power of two not above 65535) and 65536.
    let mut pred = predictor(u16::MAX);
    let idr = pred.request_keyframe((), frame_meta(0), true).unwrap();
    assert_eq!((idr.sps.max_frame_num(), idr.sps.max_pic_order_cnt_lsb()), (32768, 65536));
    pred.counter = 40_000;
    let p = pred.request_interframe((), frame_meta(40_000)).unwrap();
    assert_eq!(p.dpb_meta.poc, 80_000, "the full POC");
    assert_eq!(p.header.pic_order_cnt_lsb, (80_000 % 65_536) as u16);
    assert_eq!(p.dpb_meta.frame_num, 40_000 % 32_768);
    assert_eq!(u32::from(p.header.frame_num), 40_000 % 32_768);
}

#[test]
fn consecutive_idrs_differ_in_idr_pic_id() {
    // A one-frame GOP: every frame is an IDR.
    let mut pred = predictor(1);
    let ids: Vec<u16> = (0..3)
        .map(|n| {
            let idr = pred.request_keyframe((), frame_meta(n), true).unwrap();
            assert!(idr.is_idr && idr.parameter_sets == ParameterSets::SpsAndPps);
            let (nal, read) = read_back_slice(&idr);
            assert_eq!(nal, (3, NaluType::SliceIdr));
            assert_eq!(read.idr_pic_id, idr.header.idr_pic_id, "idr_pic_id read back");
            idr.header.idr_pic_id
        })
        .collect();
    assert_eq!(ids, [0, 1, 2], "H.264 7.4.3: two IDR access units in a row differ in idr_pic_id");
}

#[test]
fn a_gop_shorter_than_sixteen_frames_states_the_smallest_counters() {
    for limit in [1u16, 2, 10, 15] {
        let mut pred = predictor(limit);
        let idr = pred.request_keyframe((), frame_meta(0), true).unwrap();
        assert_eq!(
            idr.sps.log2_max_frame_num_minus4, 0,
            "GOP {limit}: MaxFrameNum is the smallest, 16"
        );
        assert_eq!(
            idr.sps.log2_max_pic_order_cnt_lsb_minus4, 0,
            "GOP {limit}: MaxPicOrderCntLsb 16"
        );
        let parsed = parse_sps(&synth_sps(&idr.sps));
        assert_eq!(synth_sps(&idr.sps), synth_sps(&parsed), "GOP {limit}: the SPS round-trips");
    }
}

/// Encodes `frames` frames through `pred` as the stateless encoder drives it (each frame's
/// reconstruction handed back before the next frame comes), handing `pred` the tunings
/// `tune(n)` names just before frame `n`. Returns one request per frame.
fn encode_through(
    pred: &mut LowDelayH264<(), ()>,
    frames: u64,
    mut tune: impl FnMut(u64) -> Option<Tunings>,
) -> Vec<BackendRequest<(), ()>> {
    let mut requests = Vec::new();
    for n in 0..frames {
        if let Some(tunings) = tune(n) {
            pred.tune(tunings).unwrap();
        }
        let mut pending = pred.new_frame((), frame_meta(n)).unwrap();
        while let Some(request) = pending.pop() {
            let recon = DpbEntry { recon_pic: (), meta: request.dpb_meta.clone() };
            requests.push(request);
            pending.extend(pred.reconstructed(recon).unwrap());
        }
    }
    assert_eq!(requests.len() as u64, frames, "one request a frame");
    requests
}

/// The NAL unit types the predictor wrote in band, in order.
fn in_band_nal_types(request: &BackendRequest<(), ()>) -> Vec<NaluType> {
    let mut cursor = Cursor::new(&request.coded_output[..]);
    std::iter::from_fn(|| Nalu::next(&mut cursor).ok().map(|nalu| nalu.header.type_)).collect()
}

#[test]
fn parameter_sets_travel_with_the_idr_and_the_first_frame_after_a_pps_change() {
    let mut pred = predictor(30);
    let changed = Tunings { min_quality: 10, max_quality: 40, ..Tunings::default() };
    let requests = encode_through(&mut pred, 4, |n| (n == 2).then(|| changed.clone()));

    assert_eq!(requests[0].parameter_sets, ParameterSets::SpsAndPps, "the IDR states both");
    assert_eq!(in_band_nal_types(&requests[0]), [NaluType::Sps, NaluType::Pps]);
    assert_eq!(requests[1].parameter_sets, ParameterSets::None, "nothing changed since the IDR");
    assert!(requests[1].coded_output.is_empty());
    // New quality bounds move the PPS's initial QP to their middle, 25, on the very frame they
    // govern; the SPS holds.
    assert_eq!(requests[2].parameter_sets, ParameterSets::Pps, "the PPS changed");
    assert_eq!(requests[2].pps.pic_init_qp_minus26, -1);
    assert_eq!(in_band_nal_types(&requests[2]), [NaluType::Pps], "in band, the PPS alone");
    assert_eq!(requests[2].sps, requests[0].sps, "the SPS holds");
    assert_eq!(requests[3].parameter_sets, ParameterSets::None, "once");

    // With the backend stating them, the request still says which go, and the predictor writes
    // none.
    let mut cfg = config(None);
    cfg.emit_parameter_sets = false;
    let mut pred: LowDelayH264<(), ()> = LowDelayH264::new(cfg, 30);
    let requests = encode_through(&mut pred, 3, |n| (n == 2).then(|| changed.clone()));
    assert_eq!(requests[0].parameter_sets, ParameterSets::SpsAndPps);
    assert_eq!(requests[2].parameter_sets, ParameterSets::Pps);
    assert!(requests.iter().all(|r| r.coded_output.is_empty()));
}

#[test]
fn a_bit_rate_tune_restates_no_parameter_set() {
    // A new bit rate reaches the driver through the per-frame rate-control parameters; it
    // changes neither parameter set, so no access unit restates one before the next IDR.
    let mut pred = predictor(30);
    let requests = encode_through(&mut pred, 31, |n| {
        (n == 5 || n == 12).then(|| Tunings {
            rate_control: RateControl::ConstantBitrate(100_000 * (n + 1)),
            ..Tunings::default()
        })
    });
    for (n, request) in requests.iter().enumerate().take(30).skip(1) {
        assert_eq!(request.parameter_sets, ParameterSets::None, "frame {n}");
        assert!(request.coded_output.is_empty(), "frame {n}");
    }
    assert_eq!(requests[5].tunings.rate_control, RateControl::ConstantBitrate(600_000));
    assert_eq!(requests[12].tunings.rate_control, RateControl::ConstantBitrate(1_300_000));
    assert!(requests[30].is_idr && requests[30].parameter_sets == ParameterSets::SpsAndPps);
}

#[test]
fn a_constant_qp_tune_takes_effect_on_the_frame_it_governs() {
    let mut cfg = config(None);
    cfg.initial_tunings.rate_control = RateControl::ConstantQuality(30);
    let mut pred: LowDelayH264<(), ()> = LowDelayH264::new(cfg, 30);
    let cq = |qp| Tunings { rate_control: RateControl::ConstantQuality(qp), ..Tunings::default() };
    let requests = encode_through(&mut pred, 8, |n| match n {
        3 => Some(cq(20)),
        6 => Some(cq(25)),
        _ => None,
    });

    // The PPS's initial QP is the constant QP: a QP that lagged one tune would have the driver
    // code frame 3 at 30.
    let init_qp: Vec<i8> = requests.iter().map(|r| r.pps.pic_init_qp_minus26 + 26).collect();
    assert_eq!(init_qp, [30, 30, 30, 20, 20, 20, 25, 25]);
    let pps_at: Vec<usize> =
        (1..requests.len()).filter(|&n| requests[n].parameter_sets == ParameterSets::Pps).collect();
    assert_eq!(pps_at, [3, 6], "each change restates the PPS once, on the frame it governs");
    assert_eq!(requests[0].parameter_sets, ParameterSets::SpsAndPps);
    assert!(requests[1..].iter().all(|r| r.parameter_sets != ParameterSets::SpsAndPps));
    assert!(requests.iter().all(|r| r.sps == requests[0].sps), "the SPS holds");
}

#[test]
fn frame_rate_tunes_inside_a_gop_keep_the_active_sps_to_the_next_idr() {
    // Two frame rates inside one GOP. The SPS states the rate in its VUI timing and holds for
    // the coded video sequence (H.264 7.4.1.2.1): no SPS of other content may appear before the
    // next IDR, which states the latest rate.
    let mut pred = predictor(30);
    let rate = |framerate| Tunings { framerate, ..Tunings::default() };
    let requests = encode_through(&mut pred, 35, |n| match n {
        5 => Some(rate(60)),
        10 => Some(rate(24)),
        _ => None,
    });

    let time_scale = |r: &BackendRequest<(), ()>| r.sps.vui_parameters.time_scale;
    assert_eq!(time_scale(&requests[0]), 60, "30 fps: two ticks a frame");
    for (n, request) in requests.iter().enumerate().take(30).skip(1) {
        assert!(!request.is_idr, "frame {n}");
        assert_eq!(request.sps, requests[0].sps, "frame {n}: the active SPS holds");
        assert_eq!(request.parameter_sets, ParameterSets::None, "frame {n}: nothing restated");
        assert!(request.coded_output.is_empty(), "frame {n}: no SPS in band");
    }
    // The encoder takes each rate at once, from the tunings the request carries.
    assert_eq!(requests[5].tunings.framerate, 60);
    assert_eq!(requests[10].tunings.framerate, 24);

    let idr = &requests[30];
    assert!(idr.is_idr && idr.parameter_sets == ParameterSets::SpsAndPps);
    assert_eq!(time_scale(idr), 48, "the next IDR states the latest rate, 24 fps");
    assert_eq!(in_band_nal_types(idr), [NaluType::Sps, NaluType::Pps]);
    assert!(requests[31..]
        .iter()
        .all(|r| r.sps == idr.sps && r.parameter_sets == ParameterSets::None));
}

#[test]
fn a_tune_before_the_first_frame_shapes_the_first_idr() {
    let mut cfg = config(None);
    cfg.initial_tunings.rate_control = RateControl::ConstantQuality(30);
    let mut pred: LowDelayH264<(), ()> = LowDelayH264::new(cfg, 30);
    let tuned = Tunings {
        rate_control: RateControl::ConstantQuality(22),
        framerate: 50,
        ..Tunings::default()
    };
    let requests = encode_through(&mut pred, 2, |n| (n == 0).then(|| tuned.clone()));
    assert_eq!(requests[0].parameter_sets, ParameterSets::SpsAndPps);
    assert_eq!(requests[0].pps.pic_init_qp_minus26 + 26, 22);
    assert_eq!(requests[0].sps.vui_parameters.time_scale, 100);
    assert_eq!(requests[1].parameter_sets, ParameterSets::None);
}
