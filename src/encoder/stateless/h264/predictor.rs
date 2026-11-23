// Copyright 2024 The ChromiumOS Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

use std::rc::Rc;

use log::trace;

use crate::codec::h264::parser::Level;
use crate::codec::h264::parser::Pps;
use crate::codec::h264::parser::PpsBuilder;
use crate::codec::h264::parser::Profile;
use crate::codec::h264::parser::SliceHeaderBuilder;
use crate::codec::h264::parser::SliceType;
use crate::codec::h264::parser::Sps;
use crate::codec::h264::parser::SpsBuilder;
use crate::codec::h264::synthesizer::Synthesizer;
use crate::encoder::stateless::h264::BackendRequest;
use crate::encoder::stateless::h264::DpbEntry;
use crate::encoder::stateless::h264::DpbEntryMeta;
use crate::encoder::stateless::h264::EncoderConfig;
use crate::encoder::stateless::h264::IsReference;
use crate::encoder::stateless::h264::ParameterSets;
use crate::encoder::stateless::predictor::LowDelay;
use crate::encoder::stateless::predictor::LowDelayDelegate;
use crate::encoder::stateless::FrameMetadata;
use crate::encoder::EncodeError;
use crate::encoder::EncodeResult;
use crate::encoder::RateControl;
use crate::encoder::Tunings;

pub(crate) const MIN_QP: u8 = 1;
pub(crate) const MAX_QP: u8 = 51;

pub(crate) struct LowDelayH264Delegate {
    /// The active SPS, the coded video sequence's: its IDR builds it, and it holds to the next.
    sps: Option<Rc<Sps>>,
    /// The active PPS.
    pps: Option<Rc<Pps>>,

    /// The PPS changed since an access unit last stated it, so the next one restates it.
    pps_changed: bool,

    /// The `idr_pic_id` of the next IDR. Two IDR access units in a row must differ in it
    /// (H.264 7.4.3), so it steps on at every IDR.
    next_idr_pic_id: u16,

    /// Whether SPS/PPS are synthesized into `coded_output` at all. When
    /// `false` (backend owns the parameter sets — e.g. Vulkan Video),
    /// `in_band_parameter_sets` writes nothing and `coded_output` stays empty
    /// of parameter sets. Mirrored from [`EncoderConfig::emit_parameter_sets`].
    emit_parameter_sets: bool,

    /// Encoder config
    config: EncoderConfig,
}

pub(crate) type LowDelayH264<Picture, Reference> = LowDelay<
    Picture,
    DpbEntry<Reference>,
    LowDelayH264Delegate,
    BackendRequest<Picture, Reference>,
>;

impl<Picture, Reference> LowDelayH264<Picture, Reference> {
    pub(super) fn new(config: EncoderConfig, limit: u16) -> Self {
        Self {
            queue: Default::default(),
            references: Default::default(),
            counter: 0,
            limit,
            tunings: config.initial_tunings.clone(),
            delegate: LowDelayH264Delegate {
                emit_parameter_sets: config.emit_parameter_sets,
                config,
                pps_changed: false,
                next_idr_pic_id: 0,
                sps: None,
                pps: None,
            },
            tunings_queue: Default::default(),
            _phantom: Default::default(),
        }
    }

    /// Starts a coded video sequence: its SPS and PPS, from the current tunings. Its IDR states
    /// both.
    fn new_sequence(&mut self) {
        trace!("beginning new sequence");
        let sps = build_sps(&self.delegate.config, self.limit, &self.tunings);
        let pps = build_pps(&sps, &self.tunings);
        self.delegate.sps = Some(sps);
        self.delegate.pps = Some(pps);
        self.delegate.pps_changed = false;
    }

    /// The current picture's DPB entry. Every frame is a short-term reference. `frame_num` counts
    /// the frames since the IDR, modulo `MaxFrameNum` (H.264 7.4.3); the POC counts their fields,
    /// two a frame, from 0 at the IDR (8.2.1.1).
    fn current_dpb_meta(&self, sps: &Sps) -> DpbEntryMeta {
        // Below the GOP length, a `u16`, so the POC fits as well.
        let count = self.counter as u32;
        DpbEntryMeta {
            poc: (count * 2) as i32,
            frame_num: count & (sps.max_frame_num() - 1),
            is_reference: IsReference::ShortTerm,
        }
    }

    /// The parameter sets the next access unit opens with: both on an IDR, else the PPS when it
    /// changed since an access unit last stated it.
    fn take_parameter_sets(&mut self, idr: bool) -> ParameterSets {
        let pps_changed = std::mem::take(&mut self.delegate.pps_changed);
        if idr {
            ParameterSets::SpsAndPps
        } else if pps_changed {
            ParameterSets::Pps
        } else {
            ParameterSets::None
        }
    }

    /// `parameter_sets` written in band, the start of the request's `coded_output`, where the
    /// predictor writes them (`emit_parameter_sets`); nothing where the backend states them.
    fn in_band_parameter_sets(
        &self,
        parameter_sets: ParameterSets,
        sps: &Sps,
        pps: &Pps,
    ) -> EncodeResult<Vec<u8>> {
        let mut out = Vec::new();
        if self.delegate.emit_parameter_sets {
            if parameter_sets.sps() {
                Synthesizer::<Sps, &mut Vec<u8>>::synthesize(3, sps, &mut out, true)?;
            }
            if parameter_sets.pps() {
                Synthesizer::<Pps, &mut Vec<u8>>::synthesize(3, pps, &mut out, true)?;
            }
        }
        Ok(out)
    }
}

/// The SPS of a coded video sequence of `limit`-frame GOPs at `tunings`.
fn build_sps(config: &EncoderConfig, limit: u16, tunings: &Tunings) -> Rc<Sps> {
    let mut sps = SpsBuilder::new().seq_parameter_set_id(0).profile_idc(config.profile);

    // H.264 Table 6-1
    sps = match config.profile {
        // 4:2:2 subsampling
        Profile::High422P => sps.chroma_format_idc(2),
        // 4:2:0 subsampling
        _ => sps.chroma_format_idc(1),
    };

    let mut sps = sps
        .level_idc(config.level)
        .max_frame_num(limit as u32)
        .pic_order_cnt_type(0)
        .max_pic_order_cnt_lsb(limit as u32 * 2)
        .max_num_ref_frames(1)
        .frame_mbs_only_flag(true)
        // H264 spec Table A-4
        .direct_8x8_inference_flag(config.level >= Level::L3)
        .resolution(config.resolution.width, config.resolution.height)
        .bit_depth_luma(8)
        .bit_depth_chroma(8)
        .aspect_ratio(1, 1)
        .timing_info(1, tunings.framerate * 2, false);

    // Thread the caller's optional CICP colour description into the VUI.
    // When absent, the VUI carries no `video_signal_type` block.
    if let Some(c) = config.color {
        sps = sps.video_signal_type(c.full_range, c.primaries, c.transfer, c.matrix);
    }

    sps.build()
}

/// The PPS over `sps` at `tunings`. Its initial QP is the constant QP, or the middle of the
/// quality bounds, within `MIN_QP..=MAX_QP`.
fn build_pps(sps: &Rc<Sps>, tunings: &Tunings) -> Rc<Pps> {
    let min_qp = tunings.min_quality.clamp(MIN_QP as u32, MAX_QP as u32);
    let max_qp = tunings.max_quality.clamp(min_qp, MAX_QP as u32);

    let init_qp = if let RateControl::ConstantQuality(init_qp) = tunings.rate_control {
        init_qp.clamp(min_qp, max_qp) as u8
    } else {
        ((min_qp + max_qp) / 2) as u8
    };

    PpsBuilder::new(Rc::clone(sps))
        .pic_parameter_set_id(0)
        .pic_init_qp(init_qp)
        .deblocking_filter_control_present_flag(true)
        .num_ref_idx_l0_default_active(1)
        // Unused, P frame relies only on list0
        .num_ref_idx_l1_default_active_minus1(0)
        .build()
}

impl<Picture, Reference>
    LowDelayDelegate<Picture, DpbEntry<Reference>, BackendRequest<Picture, Reference>>
    for LowDelayH264<Picture, Reference>
{
    fn request_keyframe(
        &mut self,
        input: Picture,
        input_meta: FrameMetadata,
        idr: bool,
    ) -> EncodeResult<BackendRequest<Picture, Reference>> {
        // A user `force_keyframe` (the FrameMetadata flag) is promoted to a true
        // IDR with a GOP reset — a seekable sync sample (fresh sequence, NAL 5,
        // frame_num/poc = 0), not a mid-stream non-IDR I-frame. The shared
        // predictor already cleared the reference list for this frame.
        let idr = idr || input_meta.force_keyframe;
        if idr {
            // Begin new sequence and start with an IDR (I frame, no references).
            self.new_sequence();
            // GOP reset: an IDR must carry frame_num = 0 / poc = 0. Rewind the
            // predictor's GOP counter so this frame is frame 0 of the new sequence
            // (the shared predictor increments it back to 1 after this request), so
            // the next natural IDR is a full `limit` frames away.
            self.counter = 0;
        }

        let sps = self.delegate.sps.clone().ok_or(EncodeError::InvalidInternalState)?;
        let pps = self.delegate.pps.clone().ok_or(EncodeError::InvalidInternalState)?;

        let dpb_meta = self.current_dpb_meta(&sps);
        let mut header = SliceHeaderBuilder::new(&pps)
            .slice_type(SliceType::I)
            .first_mb_in_slice(0)
            .pic_order_cnt_lsb(pic_order_cnt_lsb(&sps, &dpb_meta))
            .build();
        header.frame_num = dpb_meta.frame_num as u16;
        if idr {
            header.idr_pic_id = self.delegate.next_idr_pic_id;
            self.delegate.next_idr_pic_id = self.delegate.next_idr_pic_id.wrapping_add(1);
        }

        let parameter_sets = self.take_parameter_sets(idr);
        let coded_output = self.in_band_parameter_sets(parameter_sets, &sps, &pps)?;

        let num_macroblocks = (sps.pic_width_in_mbs_minus1 as usize + 1)
            * (sps.pic_height_in_map_units_minus1 as usize + 1);

        let request = BackendRequest {
            sps,
            pps,
            header,
            input,
            input_meta,
            dpb_meta,
            // This frame is IDR, therefore it has no references
            ref_list_0: vec![],
            ref_list_1: vec![],

            // I frame is every `self.limit` is requested
            intra_period: self.limit as u32,
            // There is no B frames between I and P frames
            ip_period: 0,

            num_macroblocks,

            is_idr: idr,
            parameter_sets,
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
        let mut ref_list_0 = vec![];

        // Use all avaiable reference frames in DPB. Their number is limited by the parameter
        for reference in self.references.iter().rev() {
            ref_list_0.push(Rc::clone(reference));
        }

        let sps = self.delegate.sps.clone().ok_or(EncodeError::InvalidInternalState)?;
        let pps = self.delegate.pps.clone().ok_or(EncodeError::InvalidInternalState)?;

        let dpb_meta = self.current_dpb_meta(&sps);
        let mut header = SliceHeaderBuilder::new(&pps)
            .slice_type(SliceType::P)
            .first_mb_in_slice(0)
            .pic_order_cnt_lsb(pic_order_cnt_lsb(&sps, &dpb_meta))
            .build();
        header.frame_num = dpb_meta.frame_num as u16;

        let parameter_sets = self.take_parameter_sets(false);
        let coded_output = self.in_band_parameter_sets(parameter_sets, &sps, &pps)?;

        let num_macroblocks = (sps.pic_width_in_mbs_minus1 as usize + 1)
            * (sps.pic_height_in_map_units_minus1 as usize + 1);

        let request = BackendRequest {
            sps,
            pps,
            header,
            input,
            input_meta,
            dpb_meta,
            ref_list_0,
            ref_list_1: vec![], // No future references

            // I frame is every `self.limit` is requested
            intra_period: self.limit as u32,
            // There is no B frames between I and P frames
            ip_period: 0,

            num_macroblocks,

            is_idr: false,
            parameter_sets,
            tunings: self.tunings.clone(),

            coded_output,
        };

        self.references.clear();

        Ok(request)
    }

    fn try_tunings(&self, _tunings: &Tunings) -> EncodeResult<()> {
        Ok(())
    }

    /// `tunings` govern the frames from the next one on; the shared predictor makes them current
    /// once this returns, so the parameter sets follow them rather than the ones they replace.
    ///
    /// The SPS holds for the coded video sequence (H.264 7.4.1.2.1), so the frame rate its VUI
    /// timing states takes effect at the next IDR, which builds the SPS from the tunings then
    /// current. Chromium's VA-API encoder (`UpdateRates`) likewise sends a rate change's SPS
    /// with the next I(DR) frame and forces none. The encoder takes the new rate at once, from
    /// the tunings every request carries.
    ///
    /// The PPS may change between pictures. Its initial QP follows the new quality bounds, or
    /// constant QP, from the next frame on, which restates it; a tune that leaves it as it is
    /// (a new bit rate) resends nothing.
    fn apply_tunings(&mut self, tunings: &Tunings) -> EncodeResult<()> {
        // Before the first IDR there is no sequence yet: that IDR builds both from `tunings`.
        let Some(sps) = self.delegate.sps.as_ref() else {
            return Ok(());
        };
        let pps = build_pps(sps, tunings);
        if self.delegate.pps.as_ref() != Some(&pps) {
            self.delegate.pps = Some(pps);
            self.delegate.pps_changed = true;
        }
        Ok(())
    }
}

/// The POC as the slice header states it, modulo `MaxPicOrderCntLsb` (H.264 7.4.3).
fn pic_order_cnt_lsb(sps: &Sps, dpb_meta: &DpbEntryMeta) -> u16 {
    (dpb_meta.poc as u32 & (sps.max_pic_order_cnt_lsb() - 1)) as u16
}
