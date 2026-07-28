// Copyright 2024 The ChromiumOS Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

use crate::codec::av1::parser::BitDepth;
use crate::codec::av1::parser::CdefParams;
use crate::codec::av1::parser::ColorConfig;
use crate::codec::av1::parser::ColorPrimaries;
use crate::codec::av1::parser::FrameHeaderObu;
use crate::codec::av1::parser::FrameType;
use crate::codec::av1::parser::MatrixCoefficients;
use crate::codec::av1::parser::TransferCharacteristics;
use crate::codec::av1::parser::ObuHeader;
use crate::codec::av1::parser::ObuType;
use crate::codec::av1::parser::OperatingPoint;
use crate::codec::av1::parser::Profile;
use crate::codec::av1::parser::QuantizationParams;
use crate::codec::av1::parser::ReferenceFrameType;
use crate::codec::av1::parser::SequenceHeaderObu;
use crate::codec::av1::parser::TemporalDelimiterObu;
use crate::codec::av1::parser::TileInfo;
use crate::codec::av1::parser::TxMode;
use crate::codec::av1::parser::MAX_NUM_OPERATING_POINTS;
use crate::codec::av1::parser::MAX_TILE_COLS;
use crate::codec::av1::parser::MAX_TILE_ROWS;
use crate::codec::av1::parser::PRIMARY_REF_NONE;
use crate::codec::av1::parser::REFS_PER_FRAME;
use crate::codec::av1::parser::SELECT_INTEGER_MV;
use crate::codec::av1::parser::SUPERRES_NUM;
use crate::codec::av1::synthesizer::Synthesizer;
use crate::encoder::stateless::av1::BackendRequest;
use crate::encoder::stateless::av1::EncoderConfig;
use crate::encoder::stateless::predictor::LowDelay;
use crate::encoder::stateless::predictor::LowDelayDelegate;
use crate::encoder::EncodeResult;
use crate::encoder::FrameMetadata;
use crate::encoder::RateControl;
use crate::encoder::Tunings;

// AV1 Spec. Dc_Qlookup max indices
pub(crate) const MIN_BASE_QINDEX: u32 = 0;
pub(crate) const MAX_BASE_QINDEX: u32 = 255;

/// One AV1 Annex A level row: the constraints a stream must satisfy to conform
/// to `seq_level_idx` (Table A.1 sizes/rates + the §A.3 tier bitrates as
/// MainMbps/HighMbps × 1e6; `high_bps == 0` = the level defines no High tier).
/// The per-level MaxTiles/MaxTileCols constraints are omitted: this predictor
/// always codes a single tile, which every level allows.
struct LevelLimits {
    seq_level_idx: u8,
    max_pic_size: u32,
    max_h_size: u32,
    max_v_size: u32,
    max_display_rate: u64,
    main_bps: u64,
    high_bps: u64,
}

/// Annex A Table A.1/A.3 — the defined levels 2.0 … 6.3 (`seq_level_idx` 2, 3,
/// 6, 7 and the 7.x block are reserved).
#[rustfmt::skip]
const LEVEL_LIMITS: &[LevelLimits] = &[
    LevelLimits { seq_level_idx: 0,  max_pic_size: 147_456,    max_h_size: 2048,  max_v_size: 1152, max_display_rate: 4_423_680,     main_bps: 1_500_000,   high_bps: 0 },
    LevelLimits { seq_level_idx: 1,  max_pic_size: 278_784,    max_h_size: 2816,  max_v_size: 1584, max_display_rate: 8_363_520,     main_bps: 3_000_000,   high_bps: 0 },
    LevelLimits { seq_level_idx: 4,  max_pic_size: 665_856,    max_h_size: 4352,  max_v_size: 2448, max_display_rate: 19_975_680,    main_bps: 6_000_000,   high_bps: 0 },
    LevelLimits { seq_level_idx: 5,  max_pic_size: 1_065_024,  max_h_size: 5504,  max_v_size: 3096, max_display_rate: 31_950_720,    main_bps: 10_000_000,  high_bps: 0 },
    LevelLimits { seq_level_idx: 8,  max_pic_size: 2_359_296,  max_h_size: 6144,  max_v_size: 3456, max_display_rate: 70_778_880,    main_bps: 12_000_000,  high_bps: 30_000_000 },
    LevelLimits { seq_level_idx: 9,  max_pic_size: 2_359_296,  max_h_size: 6144,  max_v_size: 3456, max_display_rate: 141_557_760,   main_bps: 20_000_000,  high_bps: 50_000_000 },
    LevelLimits { seq_level_idx: 12, max_pic_size: 8_912_896,  max_h_size: 8192,  max_v_size: 4352, max_display_rate: 267_386_880,   main_bps: 30_000_000,  high_bps: 100_000_000 },
    LevelLimits { seq_level_idx: 13, max_pic_size: 8_912_896,  max_h_size: 8192,  max_v_size: 4352, max_display_rate: 534_773_760,   main_bps: 40_000_000,  high_bps: 160_000_000 },
    LevelLimits { seq_level_idx: 14, max_pic_size: 8_912_896,  max_h_size: 8192,  max_v_size: 4352, max_display_rate: 1_069_547_520, main_bps: 60_000_000,  high_bps: 240_000_000 },
    LevelLimits { seq_level_idx: 15, max_pic_size: 8_912_896,  max_h_size: 8192,  max_v_size: 4352, max_display_rate: 1_069_547_520, main_bps: 60_000_000,  high_bps: 240_000_000 },
    LevelLimits { seq_level_idx: 16, max_pic_size: 35_651_584, max_h_size: 16384, max_v_size: 8704, max_display_rate: 1_069_547_520, main_bps: 60_000_000,  high_bps: 240_000_000 },
    LevelLimits { seq_level_idx: 17, max_pic_size: 35_651_584, max_h_size: 16384, max_v_size: 8704, max_display_rate: 2_139_095_040, main_bps: 100_000_000, high_bps: 480_000_000 },
    LevelLimits { seq_level_idx: 18, max_pic_size: 35_651_584, max_h_size: 16384, max_v_size: 8704, max_display_rate: 4_278_190_080, main_bps: 160_000_000, high_bps: 800_000_000 },
    LevelLimits { seq_level_idx: 19, max_pic_size: 35_651_584, max_h_size: 16384, max_v_size: 8704, max_display_rate: 4_278_190_080, main_bps: 160_000_000, high_bps: 800_000_000 },
];

/// `seq_level_idx` 23 = level 7.3 — the "conforms to no defined level"
/// declaration (with High tier), matching the convention other encoders use
/// when a stream exceeds every Annex A row.
const LEVEL_7_3: u8 = 23;

/// Pick the lowest defined level (and its tier) the configured stream conforms
/// to: `pic_size`/width/height against the Table A.1 maxima, `pic_size × fps`
/// against MaxDisplayRate, and `target_bps` against the tier bitrate caps (Main
/// preferred over High; `target_bps == 0` = no bitrate target, e.g. constant
/// quality, so the bitrate constraint is skipped). Returns
/// `(seq_level_idx, seq_tier)`; High tier only exists for `seq_level_idx > 7`,
/// so the result always satisfies the synthesizer's "`seq_tier == 0` unless
/// `seq_level_idx > 7`" rule.
pub(crate) fn guess_level(width: u32, height: u32, framerate: u32, target_bps: u64) -> (u8, u8) {
    let pic_size = u64::from(width) * u64::from(height);
    for l in LEVEL_LIMITS {
        if pic_size > u64::from(l.max_pic_size)
            || width > l.max_h_size
            || height > l.max_v_size
            || pic_size * u64::from(framerate) > l.max_display_rate
        {
            continue;
        }
        if target_bps == 0 || target_bps <= l.main_bps {
            return (l.seq_level_idx, 0);
        }
        if l.high_bps != 0 && target_bps <= l.high_bps {
            return (l.seq_level_idx, 1);
        }
    }
    (LEVEL_7_3, 1)
}

pub(crate) struct LowDelayAV1Delegate {
    /// Current sequence header obu
    sequence: SequenceHeaderObu,

    /// Encoder config
    config: EncoderConfig,
}

pub(crate) type LowDelayAV1<Picture, Reference> =
    LowDelay<Picture, Reference, LowDelayAV1Delegate, BackendRequest<Picture, Reference>>;

impl<Picture, Reference> LowDelayAV1<Picture, Reference> {
    pub fn new(config: EncoderConfig, limit: u16) -> Self {
        Self {
            queue: Default::default(),
            references: Default::default(),
            counter: 0,
            limit,
            tunings: config.initial_tunings.clone(),
            delegate: LowDelayAV1Delegate {
                sequence: Self::create_sequence_header(&config),
                config,
            },
            tunings_queue: Default::default(),
            _phantom: Default::default(),
        }
    }

    fn create_sequence_header(config: &EncoderConfig) -> SequenceHeaderObu {
        let width = config.resolution.width;
        let height = config.resolution.height;

        // Optional CICP colour description in `color_config`. When absent,
        // `color_description_present_flag == 0` with primaries/transfer/matrix
        // left Unspecified — the exact combination the synthesizer requires when
        // the flag is 0, so 8-bit/10-bit streams without colour stay byte-identical.
        let (
            color_description_present_flag,
            color_primaries,
            transfer_characteristics,
            matrix_coefficients,
            color_range,
        ) = match config.color {
            Some(c) => (
                true,
                ColorPrimaries::try_from(c.primaries as u32).unwrap_or(ColorPrimaries::Unspecified),
                TransferCharacteristics::try_from(c.transfer as u32)
                    .unwrap_or(TransferCharacteristics::Unspecified),
                MatrixCoefficients::try_from(c.matrix as u32)
                    .unwrap_or(MatrixCoefficients::Unspecified),
                c.full_range,
            ),
            None => (
                false,
                ColorPrimaries::Unspecified,
                TransferCharacteristics::Unspecified,
                MatrixCoefficients::Unspecified,
                false,
            ),
        };

        SequenceHeaderObu {
            obu_header: ObuHeader {
                obu_type: ObuType::SequenceHeader,
                extension_flag: false,
                has_size_field: true,
                temporal_id: 0,
                spatial_id: 0,
            },

            seq_profile: Profile::Profile0,
            num_planes: 3,

            enable_order_hint: true,
            order_hint_bits: 8,
            order_hint_bits_minus_1: 8 - 1,

            // Use maximum size (16 bits)
            frame_width_bits_minus_1: (1 << 4) - 1,
            frame_height_bits_minus_1: (1 << 4) - 1,

            // Current resolution is the maximum resolution
            max_frame_width_minus_1: (width - 1) as u16,
            max_frame_height_minus_1: (height - 1) as u16,

            seq_force_integer_mv: SELECT_INTEGER_MV as u32,

            operating_points: {
                let mut ops: [OperatingPoint; MAX_NUM_OPERATING_POINTS] = Default::default();
                ops[0].idc = 0;
                // Declared conformance: the lowest Annex A level the configured
                // stream fits. CBR/VBR conform at the (peak) target bitrate; a
                // constant-quality stream has no bitrate target, so only the
                // size/rate constraints apply.
                let target_bps = match config.initial_tunings.rate_control {
                    RateControl::ConstantBitrate(bps) => bps,
                    RateControl::VariableBitrate { max_bitrate, .. } => max_bitrate,
                    RateControl::ConstantQuality(_) => 0,
                };
                let (seq_level_idx, seq_tier) =
                    guess_level(width, height, config.initial_tunings.framerate, target_bps);
                ops[0].seq_level_idx = seq_level_idx;
                ops[0].seq_tier = seq_tier;
                ops
            },

            bit_depth: config.bit_depth,
            color_config: ColorConfig {
                // YUV 4:2:0 8-bit or 10-bit
                high_bitdepth: config.bit_depth == BitDepth::Depth10,
                mono_chrome: false,
                subsampling_x: true,
                subsampling_y: true,
                color_description_present_flag,
                color_primaries,
                transfer_characteristics,
                matrix_coefficients,
                color_range,
                ..Default::default()
            },

            ..Default::default()
        }
    }

    fn create_temporal_delimiter() -> TemporalDelimiterObu {
        TemporalDelimiterObu {
            obu_header: ObuHeader {
                obu_type: ObuType::TemporalDelimiter,
                extension_flag: false,
                has_size_field: true,
                temporal_id: 0,
                spatial_id: 0,
            },
        }
    }

    fn create_frame_header(&self, frame_type: FrameType) -> EncodeResult<FrameHeaderObu> {
        let width = self.delegate.config.resolution.width;
        let height = self.delegate.config.resolution.height;

        // Superblock size
        let sb_size = if self.delegate.sequence.use_128x128_superblock { 128 } else { 64 };

        // Use frame counter for order hinting
        let order_hint_mask = (1 << self.delegate.sequence.order_hint_bits) - 1;
        let order_hint = (self.counter & order_hint_mask) as u32;

        // Clamp tunings's quality range to the AV1 base Q index domain. Raising
        // the upper bound to at least the lower one keeps `clamp` total even for
        // a misconfigured `min_quality > MAX_BASE_QINDEX`.
        let min_q_idx = self.tunings.min_quality.clamp(MIN_BASE_QINDEX, MAX_BASE_QINDEX);
        let max_q_idx = self.tunings.max_quality.clamp(min_q_idx, MAX_BASE_QINDEX);

        let base_q_idx = match self.tunings.rate_control {
            // Clamp the requested constant Q index into the allowed range
            RateControl::ConstantQuality(base_q_idx) => base_q_idx.clamp(min_q_idx, max_q_idx),
            // CBR/VBR: the driver's rate controller picks the effective per-frame
            // quantizer within [min_q_idx, max_q_idx]; seed `base_q_idx` with the
            // midpoint of that range, as the H.264/VP9/HEVC predictors do.
            _ => (min_q_idx + max_q_idx) / 2,
        };

        // Set the frame size in superblocks for the only tile
        let mut width_in_sbs_minus_1 = [0u32; MAX_TILE_COLS];
        width_in_sbs_minus_1[0] = ((width + sb_size - 1) / sb_size) - 1;

        let mut height_in_sbs_minus_1 = [0u32; MAX_TILE_ROWS];
        height_in_sbs_minus_1[0] = ((height + sb_size - 1) / sb_size) - 1;

        Ok(FrameHeaderObu {
            obu_header: ObuHeader {
                obu_type: ObuType::FrameHeader,
                extension_flag: false,
                has_size_field: true,
                temporal_id: 0,
                spatial_id: 0,
            },
            show_frame: true,
            showable_frame: !matches!(frame_type, FrameType::KeyFrame),
            frame_type,
            frame_is_intra: matches!(frame_type, FrameType::KeyFrame | FrameType::IntraOnlyFrame),
            primary_ref_frame: PRIMARY_REF_NONE,
            refresh_frame_flags: if matches!(frame_type, FrameType::KeyFrame) {
                0xff
            } else {
                0x01
            },

            // Use error resilient mode, and provide the order hints for referencing frame ie. just
            // previous frame
            error_resilient_mode: true,
            order_hint,
            ref_order_hint: [0, 0, 0, 0, 0, 0, 0, 0],

            reduced_tx_set: true,
            tx_mode_select: 1,
            tx_mode: TxMode::Select,

            // Provide the Q index from config
            quantization_params: QuantizationParams { base_q_idx, ..Default::default() },

            // Use single tile for now
            tile_info: TileInfo {
                uniform_tile_spacing_flag: true,
                tile_cols: 1,
                tile_rows: 1,
                tile_cols_log2: 0,
                tile_rows_log2: 0,
                width_in_sbs_minus_1,
                height_in_sbs_minus_1,
                ..Default::default()
            },

            // CDEF is not used currently, use default value to keep Synthesizer happy
            cdef_params: CdefParams { cdef_damping: 3, ..Default::default() },

            // No superres
            superres_denom: SUPERRES_NUM as u32,
            upscaled_width: width,
            frame_width: width,
            frame_height: height,
            render_width: width,
            render_height: height,

            ..Default::default()
        })
    }
}

impl<Picture, Reference> LowDelayDelegate<Picture, Reference, BackendRequest<Picture, Reference>>
    for LowDelayAV1<Picture, Reference>
{
    fn request_keyframe(
        &mut self,
        input: Picture,
        input_meta: FrameMetadata,
        idr: bool,
    ) -> EncodeResult<BackendRequest<Picture, Reference>> {
        log::trace!("Requested keyframe timestamp={}", input_meta.timestamp);

        let temporal_delim = Self::create_temporal_delimiter();
        let sequence = self.delegate.sequence.clone();
        let frame = self.create_frame_header(FrameType::KeyFrame)?;

        // This is intra frame, so there is no references
        let references = [None, None, None, None, None, None, None];
        let ref_frame_ctrl_l0 = [ReferenceFrameType::Intra; REFS_PER_FRAME];
        let ref_frame_ctrl_l1 = [ReferenceFrameType::Intra; REFS_PER_FRAME];

        let mut coded_output = Vec::new();

        // Output Temporal Delimiter, Sequence Header and Frame Header OBUs to bitstream
        Synthesizer::<'_, TemporalDelimiterObu, _>::synthesize(&temporal_delim, &mut coded_output)?;
        if idr {
            Synthesizer::<'_, SequenceHeaderObu, _>::synthesize(&sequence, &mut coded_output)?;
        }
        Synthesizer::<'_, FrameHeaderObu, _>::synthesize(&frame, &sequence, &mut coded_output)?;

        let request = BackendRequest {
            sequence,
            frame,
            input,
            input_meta,
            references,
            ref_frame_ctrl_l0,
            ref_frame_ctrl_l1,
            intra_period: self.limit as u32,
            ip_period: 1,
            tunings: self.tunings.clone(),
            coded_output,
        };

        Ok(request)
    }

    fn request_interframe(
        &mut self,
        input: Picture,
        input_meta: FrameMetadata,
    ) -> EncodeResult<BackendRequest<Picture, Reference>> {
        log::trace!("Requested interframe timestamp={}", input_meta.timestamp);

        let temporal_delim = Self::create_temporal_delimiter();
        let sequence = self.delegate.sequence.clone();
        let mut frame = self.create_frame_header(FrameType::InterFrame)?;

        // Use previous frame as last frame reference
        let references = [self.references.front().cloned(), None, None, None, None, None, None];

        let order_hint_mask = (1 << self.delegate.sequence.order_hint_bits) - 1;
        let mut ref_frame_ctrl_l0 = [ReferenceFrameType::Intra; REFS_PER_FRAME];
        let ref_frame_ctrl_l1 = [ReferenceFrameType::Intra; REFS_PER_FRAME];

        // Enable previous frame as reference
        ref_frame_ctrl_l0[0] = ReferenceFrameType::Last;
        frame.ref_frame_idx[0] = 0;
        frame.last_frame_idx = 0;
        frame.ref_order_hint[0] = ((self.counter - 1) & order_hint_mask) as u32;

        let mut coded_output = Vec::new();

        // Output Temporal Delimiter and Frame Header OBUs to bitstream, marking next frame
        Synthesizer::<'_, TemporalDelimiterObu, _>::synthesize(&temporal_delim, &mut coded_output)?;
        Synthesizer::<'_, FrameHeaderObu, _>::synthesize(&frame, &sequence, &mut coded_output)?;

        let request = BackendRequest {
            sequence,
            frame,
            input,
            input_meta,
            references,
            ref_frame_ctrl_l0,
            ref_frame_ctrl_l1,
            intra_period: self.limit as u32,
            ip_period: 1,
            tunings: self.tunings.clone(),
            coded_output,
        };

        self.references.clear();

        Ok(request)
    }

    fn try_tunings(&self, _tunings: &Tunings) -> EncodeResult<()> {
        Ok(())
    }

    fn apply_tunings(&mut self, _tunings: &Tunings) -> EncodeResult<()> {
        Ok(())
    }
}
