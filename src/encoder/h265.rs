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

/// The luma coding-tree / coding-block / transform-block geometry the SPS codes,
/// in luma samples.
///
/// The backend owns this: an encoder backend knows which block sizes its hardware
/// accepts and builds it with [`BlockSizes::from_caps`] from its capability query.
/// Backends without such a query use [`BlockSizes::CONSERVATIVE`], the 32-CTB /
/// 8-CU / 4..=32-TB geometry every HEVC encoder accepts.
///
/// The fields are private because they are interdependent: every constructor
/// normalises them so the SPS the predictor derives satisfies the block-size
/// constraints of 7.4.3.2.1 — `CtbSizeY` is 16, 32 or 64;
/// `MinTbLog2SizeY < MinCbLog2SizeY <= CtbLog2SizeY`; and
/// `MaxTbLog2SizeY <= Min(CtbLog2SizeY, 5)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlockSizes {
    ctb_size: u32,
    min_cb_size: u32,
    min_tb_size: u32,
    max_tb_size: u32,
}

impl BlockSizes {
    /// `CtbSizeY` 32, `MinCbSizeY` 8, transform blocks 4..=32 — the geometry every
    /// HEVC encoder accepts, for backends that cannot (yet) ask the hardware what
    /// it supports.
    pub const CONSERVATIVE: Self =
        Self { ctb_size: 32, min_cb_size: 8, min_tb_size: 4, max_tb_size: 32 };

    /// The geometry to code for hardware that supports coding-tree blocks up to
    /// `max_ctb_size` and transform blocks over `min_tb_size..=max_tb_size` (all in
    /// luma samples). Out-of-spec or unadvertised sizes are rounded down to the
    /// nearest legal one, so a driver that reports nothing lands on the smallest
    /// legal geometry rather than an SPS no decoder accepts.
    ///
    /// `MinCbSizeY` is **16 whenever the CTB is at least 32**. An 8x8 min CU adds a
    /// quad-tree depth the encoder must search for every CTB, which costs a large
    /// fraction of the frame rate at the quality levels this encoder drives, while
    /// H.264 — whose macroblocks are 16x16 — pays nothing for it. FFmpeg measured
    /// 1080p HEVC at ~200-317 fps against H.264's 288-359 with the 8x8 min CU, and
    /// parity (289/325/358) after the switch (FFmpeg 426841da9d9). A 16-sample CTB
    /// keeps the 8x8 min CU: `MinCbLog2SizeY <= CtbLog2SizeY`.
    pub fn from_caps(max_ctb_size: u32, min_tb_size: u32, max_tb_size: u32) -> Self {
        let ctb_size = floor_pow2(max_ctb_size, 16, 64);
        // MaxTbLog2SizeY <= Min(CtbLog2SizeY, 5): the transform never exceeds the CTB
        // (nor 32, the largest transform HEVC defines).
        let max_tb_size = floor_pow2(max_tb_size, 4, 32).min(ctb_size);
        let min_tb_size = floor_pow2(min_tb_size, 4, 32).min(max_tb_size);
        let min_cb_size = if ctb_size >= 32 { 16 } else { 8 };
        // MinTbLog2SizeY < MinCbLog2SizeY. Unreachable on real hardware (HEVC
        // mandates the 4x4 transform, so `min_tb_size` is 4), but keep the
        // constructor total rather than emitting an SPS the parser rejects.
        let min_tb_size = min_tb_size.min(min_cb_size / 2);
        Self { ctb_size, min_cb_size, min_tb_size, max_tb_size }
    }

    /// `CtbSizeY`.
    pub fn ctb_size(&self) -> u32 {
        self.ctb_size
    }

    /// `MinCbSizeY` — also the alignment the coded picture dimensions are padded
    /// to (the conformance window crops the difference back to the display size).
    pub fn min_cb_size(&self) -> u32 {
        self.min_cb_size
    }

    /// The smallest luma transform block.
    pub fn min_tb_size(&self) -> u32 {
        self.min_tb_size
    }

    /// The largest luma transform block.
    pub fn max_tb_size(&self) -> u32 {
        self.max_tb_size
    }

    /// `CtbLog2SizeY`.
    pub fn ctb_log2(&self) -> u32 {
        self.ctb_size.ilog2()
    }

    /// `MinCbLog2SizeY`.
    pub fn min_cb_log2(&self) -> u32 {
        self.min_cb_size.ilog2()
    }

    /// `MaxTbLog2SizeY`.
    pub fn max_tb_log2(&self) -> u32 {
        self.max_tb_size.ilog2()
    }

    /// The SPS `log2_min_luma_coding_block_size_minus3`.
    pub fn log2_min_cb_minus3(&self) -> u8 {
        (self.min_cb_log2() - 3) as u8
    }

    /// The SPS `log2_diff_max_min_luma_coding_block_size`.
    pub fn log2_diff_max_min_cb(&self) -> u8 {
        (self.ctb_log2() - self.min_cb_log2()) as u8
    }

    /// The SPS `log2_min_luma_transform_block_size_minus2`.
    pub fn log2_min_tb_minus2(&self) -> u8 {
        (self.min_tb_size.ilog2() - 2) as u8
    }

    /// The SPS `log2_diff_max_min_luma_transform_block_size`.
    pub fn log2_diff_max_min_tb(&self) -> u8 {
        (self.max_tb_log2() - self.min_tb_size.ilog2()) as u8
    }

    /// The largest `max_transform_hierarchy_depth_{inter,intra}` this geometry
    /// admits: `CtbLog2SizeY - MinTbLog2SizeY` (7.4.3.2.1).
    pub fn max_transform_hierarchy_depth(&self) -> u8 {
        (self.ctb_log2() - self.min_tb_size.ilog2()) as u8
    }
}

impl Default for BlockSizes {
    fn default() -> Self {
        Self::CONSERVATIVE
    }
}

/// `v` clamped to `[lo, hi]` and then rounded **down** to a power of two. `lo` must
/// be a power of two, which makes the result one too.
fn floor_pow2(v: u32, lo: u32, hi: u32) -> u32 {
    1 << v.clamp(lo, hi).ilog2()
}

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
    /// (the VUI stays absent).
    pub color: Option<EncoderColorInfo>,
    /// The block geometry the SPS codes — the backend's capability-derived
    /// [`BlockSizes`], or [`BlockSizes::CONSERVATIVE`] where the backend has no
    /// block-size query. Also fixes the coded-dimension alignment
    /// ([`BlockSizes::min_cb_size`]), so a backend that sizes its own surfaces
    /// from the coded picture must use the same value it passes here.
    pub block_sizes: BlockSizes,
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
            block_sizes: BlockSizes::CONSERVATIVE,
        }
    }
}
