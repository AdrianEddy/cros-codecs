// Copyright 2024 The ChromiumOS Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

//! Host (GPU-independent) unit tests for the H.264 encoder's colour
//! threading: the predictor writes the caller's CICP colour description into the
//! SPS VUI (`video_signal_type` + `colour_description`), round-tripped through
//! the synthesizer and the parser.

use std::io::Cursor;
use std::rc::Rc;

use crate::codec::h264::parser::Level;
use crate::codec::h264::parser::Nalu;
use crate::codec::h264::parser::NaluType;
use crate::codec::h264::parser::Parser;
use crate::codec::h264::parser::Profile;
use crate::codec::h264::parser::Sps;
use crate::codec::h264::synthesizer::Synthesizer;
use crate::encoder::h264::EncoderConfig;
use crate::encoder::stateless::h264::predictor::LowDelayH264;
use crate::encoder::stateless::predictor::LowDelayDelegate;
use crate::encoder::stateless::FrameMetadata;
use crate::encoder::EncoderColorInfo;
use crate::encoder::PredictionStructure;
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
