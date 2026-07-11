// Copyright 2024 The ChromiumOS Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

//! Host (GPU-independent) unit tests for the AV1 encoder's 10-bit selection and
//! colour threading: the predictor sets `bit_depth`/`high_bitdepth` for
//! Profile0 10-bit and writes the caller's CICP `color_config` into the sequence
//! header OBU, round-tripped through the synthesizer and the parser.
//!
//! Also covers the bitrate-controlled (CBR/VBR) frame-header path: the
//! predictor seeds `base_q_idx` with the midpoint of the allowed quality range
//! instead of rejecting the request (which used to consume the input frame and
//! desynchronize the encoder's `predictor_frame_count`).

use std::rc::Rc;

use crate::codec::av1::parser::BitDepth;
use crate::codec::av1::parser::ColorPrimaries;
use crate::codec::av1::parser::FrameType;
use crate::codec::av1::parser::MatrixCoefficients;
use crate::codec::av1::parser::ObuAction;
use crate::codec::av1::parser::ParsedObu;
use crate::codec::av1::parser::Parser;
use crate::codec::av1::parser::Profile;
use crate::codec::av1::parser::SequenceHeaderObu;
use crate::codec::av1::parser::TransferCharacteristics;
use crate::codec::av1::synthesizer::Synthesizer;
use crate::encoder::av1::EncoderConfig;
use crate::encoder::stateless::av1::predictor::LowDelayAV1;
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

fn config_with_tunings(
    bit_depth: BitDepth,
    color: Option<EncoderColorInfo>,
    initial_tunings: Tunings,
) -> EncoderConfig {
    EncoderConfig {
        profile: Profile::Profile0,
        bit_depth,
        resolution: Resolution { width: 1920, height: 1080 },
        pred_structure: PredictionStructure::LowDelay { limit: 1024 },
        initial_tunings,
        color,
    }
}

fn config(bit_depth: BitDepth, color: Option<EncoderColorInfo>) -> EncoderConfig {
    config_with_tunings(
        bit_depth,
        color,
        Tunings { rate_control: RateControl::ConstantQuality(128), ..Default::default() },
    )
}

fn frame_meta(timestamp: u64) -> FrameMetadata {
    FrameMetadata {
        timestamp,
        layout: FrameLayout {
            format: (Fourcc::from(b"P010"), 0),
            size: Resolution { width: 1920, height: 1080 },
            planes: vec![],
        },
        force_keyframe: false,
    }
}

/// Drive the predictor's keyframe path and return the sequence header it built.
fn keyframe_sequence(bit_depth: BitDepth, color: Option<EncoderColorInfo>) -> SequenceHeaderObu {
    let mut pred: LowDelayAV1<(), ()> = LowDelayAV1::new(config(bit_depth, color), 1024);
    let req = pred.request_keyframe((), frame_meta(0), true).unwrap();
    req.sequence
}

/// Synthesize the sequence header OBU and parse it back through the AV1 parser.
fn round_trip(seq: &SequenceHeaderObu) -> Rc<SequenceHeaderObu> {
    let mut buf = Vec::new();
    Synthesizer::<'_, SequenceHeaderObu, _>::synthesize(seq, &mut buf).unwrap();

    let mut parser = Parser::default();
    let obu = match parser.read_obu(&buf).unwrap() {
        ObuAction::Process(obu) => obu,
        ObuAction::Drop(_) => panic!("sequence header OBU was dropped"),
    };
    match parser.parse_obu(obu).unwrap() {
        ParsedObu::SequenceHeader(s) => s,
        other => panic!("expected a sequence header, got {:?}", other.obu_type()),
    }
}

#[test]
fn profile0_10bit_selects_high_bitdepth() {
    let seq = keyframe_sequence(BitDepth::Depth10, None);
    assert_eq!(seq.seq_profile, Profile::Profile0);
    assert_eq!(seq.bit_depth, BitDepth::Depth10);
    assert!(seq.color_config.high_bitdepth, "10-bit ⇒ high_bitdepth");
    // No colour requested ⇒ description absent, primaries left Unspecified.
    assert!(!seq.color_config.color_description_present_flag);

    let parsed = round_trip(&seq);
    assert_eq!(parsed.bit_depth, BitDepth::Depth10, "10-bit survives round-trip");
    assert!(!parsed.color_config.color_description_present_flag);
}

#[test]
fn profile0_8bit_stays_8bit_no_colour() {
    let seq = keyframe_sequence(BitDepth::Depth8, None);
    assert_eq!(seq.bit_depth, BitDepth::Depth8);
    assert!(!seq.color_config.high_bitdepth, "8-bit ⇒ no high_bitdepth");

    let parsed = round_trip(&seq);
    assert_eq!(parsed.bit_depth, BitDepth::Depth8);
    assert!(!parsed.color_config.color_description_present_flag, "colour absent by default");
}

#[test]
fn cicp_color_config_survives_round_trip_10bit() {
    // BT.2020 NCL / PQ, limited range — HDR10 in a Profile0 10-bit stream.
    let color = EncoderColorInfo { primaries: 9, transfer: 16, matrix: 9, full_range: false };
    let seq = keyframe_sequence(BitDepth::Depth10, Some(color));

    let cc = &seq.color_config;
    assert!(cc.color_description_present_flag);
    assert_eq!(cc.color_primaries, ColorPrimaries::Bt2020);
    assert_eq!(cc.transfer_characteristics, TransferCharacteristics::Smpte2084);
    assert_eq!(cc.matrix_coefficients, MatrixCoefficients::Bt2020Ncl);
    assert!(!cc.color_range, "limited range");

    let parsed = round_trip(&seq);
    let pcc = &parsed.color_config;
    assert!(pcc.color_description_present_flag, "colour survives synth+parse");
    assert_eq!(pcc.color_primaries, ColorPrimaries::Bt2020);
    assert_eq!(pcc.transfer_characteristics, TransferCharacteristics::Smpte2084);
    assert_eq!(pcc.matrix_coefficients, MatrixCoefficients::Bt2020Ncl);
    assert!(!pcc.color_range);
    assert_eq!(parsed.bit_depth, BitDepth::Depth10);
}

#[test]
fn cicp_full_range_bt709_survives_round_trip_8bit() {
    // BT.709 limited→full range flag, 8-bit Profile0.
    let color = EncoderColorInfo { primaries: 1, transfer: 1, matrix: 1, full_range: true };
    let seq = keyframe_sequence(BitDepth::Depth8, Some(color));

    let parsed = round_trip(&seq);
    let pcc = &parsed.color_config;
    assert!(pcc.color_description_present_flag);
    assert_eq!(pcc.color_primaries, ColorPrimaries::Bt709);
    assert_eq!(pcc.transfer_characteristics, TransferCharacteristics::Bt709);
    assert_eq!(pcc.matrix_coefficients, MatrixCoefficients::Bt709);
    assert!(pcc.color_range, "full range");
}

fn predictor_with_rc(rate_control: RateControl, tunings: Tunings) -> LowDelayAV1<(), ()> {
    let tunings = Tunings { rate_control, ..tunings };
    LowDelayAV1::new(config_with_tunings(BitDepth::Depth8, None, tunings), 1024)
}

#[test]
fn cbr_seeds_midpoint_base_q_idx() {
    let mut pred =
        predictor_with_rc(RateControl::ConstantBitrate(4_000_000), Tunings::default());

    // Default quality bounds span the full AV1 domain [0, 255] → midpoint 127.
    let req = pred.request_keyframe((), frame_meta(0), true).unwrap();
    assert_eq!(req.frame.frame_type, FrameType::KeyFrame);
    assert_eq!(req.frame.quantization_params.base_q_idx, 127);
}

#[test]
fn vbr_seeds_midpoint_base_q_idx() {
    let mut pred = predictor_with_rc(
        RateControl::VariableBitrate { avg_bitrate: 4_000_000, max_bitrate: 8_000_000 },
        Tunings::default(),
    );

    let req = pred.request_keyframe((), frame_meta(0), true).unwrap();
    assert_eq!(req.frame.quantization_params.base_q_idx, 127);

    // The midpoint respects the tunings' quality bounds.
    let mut pred = predictor_with_rc(
        RateControl::VariableBitrate { avg_bitrate: 4_000_000, max_bitrate: 8_000_000 },
        Tunings { min_quality: 100, max_quality: 200, ..Default::default() },
    );

    let req = pred.request_keyframe((), frame_meta(0), true).unwrap();
    assert_eq!(req.frame.quantization_params.base_q_idx, (100 + 200) / 2);
}

#[test]
fn constant_quality_still_clamps_into_bounds() {
    let mut pred = predictor_with_rc(RateControl::ConstantQuality(128), Tunings::default());
    let req = pred.request_keyframe((), frame_meta(0), true).unwrap();
    assert_eq!(req.frame.quantization_params.base_q_idx, 128);

    // A request above the AV1 domain clamps to MAX_BASE_QINDEX.
    let mut pred = predictor_with_rc(RateControl::ConstantQuality(300), Tunings::default());
    let req = pred.request_keyframe((), frame_meta(0), true).unwrap();
    assert_eq!(req.frame.quantization_params.base_q_idx, 255);
}

/// Quality bounds entirely above the AV1 base Q index domain must not panic
/// (`clamp` stays total) and must stay within the domain, for every
/// rate-control variant.
#[test]
fn out_of_domain_quality_bounds_stay_total() {
    for rc in [
        RateControl::ConstantQuality(500),
        RateControl::ConstantBitrate(4_000_000),
        RateControl::VariableBitrate { avg_bitrate: 4_000_000, max_bitrate: 8_000_000 },
    ] {
        let mut pred = predictor_with_rc(
            rc,
            Tunings { min_quality: 300, max_quality: 400, ..Default::default() },
        );
        let req = pred.request_keyframe((), frame_meta(0), true).unwrap();
        assert_eq!(req.frame.quantization_params.base_q_idx, 255);
    }
}

/// Regression test for the P1 failure-state bug: a CBR/VBR stream used to fail
/// `create_frame_header` with `EncodeError::Unsupported` *after* the input was
/// consumed, leaving the encoder's frame accounting desynchronized (a later
/// drain would report `InvalidInternalState`). Drive the full [`Predictor`]
/// interface and check every enqueued frame comes back out as a request.
#[test]
fn bitrate_controlled_flow_keeps_predictor_state_consistent() {
    for rc in [
        RateControl::ConstantBitrate(4_000_000),
        RateControl::VariableBitrate { avg_bitrate: 4_000_000, max_bitrate: 8_000_000 },
    ] {
        let mut pred = predictor_with_rc(rc, Tunings::default());

        // First frame produces a keyframe request right away.
        let requests = pred.new_frame((), frame_meta(0)).unwrap();
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].frame.frame_type, FrameType::KeyFrame);
        assert_eq!(requests[0].frame.quantization_params.base_q_idx, 127);

        // Second frame waits for the keyframe's reconstructed reference.
        let requests = pred.new_frame((), frame_meta(1)).unwrap();
        assert!(requests.is_empty());

        // The reconstructed keyframe unblocks the interframe request.
        let requests = pred.reconstructed(()).unwrap();
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].frame.frame_type, FrameType::InterFrame);
        assert_eq!(requests[0].frame.quantization_params.base_q_idx, 127);

        // Both inputs came back out: the predictor holds no leftover frames, so
        // the encoder's `predictor_frame_count` bookkeeping stays balanced.
        let requests = pred.reconstructed(()).unwrap();
        assert!(requests.is_empty());
    }
}
