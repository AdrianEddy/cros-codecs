// Copyright 2024 The ChromiumOS Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

use crate::codec::h265::parser::Level;
use crate::codec::h265::parser::Profile;
use crate::encoder::EncoderColorInfo;
use crate::encoder::PredictionStructure;
use crate::encoder::Tunings;
use crate::Resolution;

pub struct H265;

#[derive(Clone)]
pub struct EncoderConfig {
    pub resolution: Resolution,
    /// General profile: `Main` (8-bit 4:2:0), `Main10` (10-bit 4:2:0), or
    /// `RangeExtensions` (Main 4:2:2 10, the only RExt arm the encoder emits).
    /// The bit depth and chroma format are derived from this (see the predictor)
    /// and the VA profile from it (see the VA-API backend `new_vaapi`).
    pub profile: Profile,
    pub level: Level,
    pub pred_structure: PredictionStructure,
    /// Initial tunings values
    pub initial_tunings: Tunings,
    /// Optional CICP colour description threaded into the SPS VUI
    /// (`colour_description` + `video_full_range_flag`). `None` ⇒ no VUI colour
    /// (byte-identical to the M7 Main encoder — VUI stays absent).
    pub color: Option<EncoderColorInfo>,
}

impl Default for EncoderConfig {
    fn default() -> Self {
        // Artificially encoder configuration with intent to be widely supported.
        Self {
            resolution: Resolution { width: 320, height: 240 },
            profile: Profile::Main,
            level: Level::L4,
            pred_structure: PredictionStructure::LowDelay { limit: 2048 },
            initial_tunings: Default::default(),
            color: None,
        }
    }
}
