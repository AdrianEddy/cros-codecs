// Copyright 2024 The ChromiumOS Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

//! Host (GPU-independent) unit tests for the AV1 encoder's 10-bit selection and
//! **W-F5** colour threading: the predictor sets `bit_depth`/`high_bitdepth` for
//! Profile0 10-bit and writes the caller's CICP `color_config` into the sequence
//! header OBU, round-tripped through the synthesizer and the parser.

use std::rc::Rc;

use crate::codec::av1::parser::BitDepth;
use crate::codec::av1::parser::ColorPrimaries;
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
use crate::encoder::EncoderColorInfo;
use crate::encoder::PredictionStructure;
use crate::encoder::RateControl;
use crate::encoder::Tunings;
use crate::Fourcc;
use crate::FrameLayout;
use crate::Resolution;

fn config(bit_depth: BitDepth, color: Option<EncoderColorInfo>) -> EncoderConfig {
    EncoderConfig {
        profile: Profile::Profile0,
        bit_depth,
        resolution: Resolution { width: 1920, height: 1080 },
        pred_structure: PredictionStructure::LowDelay { limit: 1024 },
        // The AV1 frame-header path derives base_q_idx from a ConstantQuality
        // rate control (it errors on a bitrate target), so the host tests use CQP.
        initial_tunings: Tunings {
            rate_control: RateControl::ConstantQuality(128),
            ..Default::default()
        },
        color,
    }
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
