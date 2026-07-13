// Copyright 2024 The ChromiumOS Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

use crate::codec::h264::parser::Level;
use crate::codec::h264::parser::Profile;
use crate::encoder::EncoderColorInfo;
use crate::encoder::PredictionStructure;
use crate::encoder::Tunings;
use crate::Resolution;

pub struct H264;

#[derive(Clone)]
pub struct EncoderConfig {
    pub resolution: Resolution,
    pub profile: Profile,
    pub level: Level,
    pub pred_structure: PredictionStructure,
    /// Initial tunings values
    pub initial_tunings: Tunings,
    /// Optional CICP colour description threaded into the SPS VUI
    /// (`colour_description` + `video_full_range_flag`). `None` ⇒ no colour is
    /// signalled.
    pub color: Option<EncoderColorInfo>,
    /// Whether the predictor synthesizes SPS/PPS NAL units into
    /// `coded_output`. Default `true` (the historical behaviour — the backend
    /// owns nothing and the parameter sets appear in-band on IDR / on change).
    /// A backend whose driver retrieves compliant parameter-set bytes itself
    /// (e.g. Vulkan Video via `vkGetEncodedVideoSessionParametersKHR`) sets
    /// this `false` so the predictor never duplicates them: `coded_output` then
    /// carries only the slice data the backend appends, and the backend prepends
    /// the driver-authoritative headers at IDR.
    pub emit_parameter_sets: bool,
}

impl Default for EncoderConfig {
    fn default() -> Self {
        // Artificially encoder configuration with intent to be widely supported.
        Self {
            resolution: Resolution { width: 320, height: 240 },
            profile: Profile::Baseline,
            level: Level::L4,
            pred_structure: PredictionStructure::LowDelay { limit: 2048 },
            initial_tunings: Default::default(),
            color: None,
            emit_parameter_sets: true,
        }
    }
}
