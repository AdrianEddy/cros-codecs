// Copyright 2024 The ChromiumOS Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

//! Stateless HEVC (H.265) encoder — the codec-generic half. Mirrors
//! [`crate::encoder::stateless::h264`]; the HEVC-specific differences are:
//!
//! - The DPB metadata keys on **POC** and the **NAL unit type** — HEVC has no
//!   `frame_num` (references are ordered by picture order count).
//! - The [`BackendRequest`] carries an [`Rc<Vps>`] alongside the SPS/PPS (HEVC
//!   adds the video parameter set), the active `NaluType` for the slice, and the
//!   per-request packed-header hint (`is_idr`) the VA-API backend uses to decide
//!   which application-packed headers to emit.
//! - The reference-picture-set is the trivial LowDelay-IPPP one (one negative
//!   reference, `delta_poc_s0_minus1 == 0`), carried in the slice header.

use std::rc::Rc;

use crate::codec::h265::parser::NaluType;
use crate::codec::h265::parser::Pps;
use crate::codec::h265::parser::SliceHeader;
use crate::codec::h265::parser::Sps;
use crate::codec::h265::parser::Vps;
use crate::encoder::h265::EncoderConfig;
use crate::encoder::h265::H265;
use crate::encoder::stateless::h265::predictor::LowDelayH265;
use crate::encoder::stateless::BackendPromise;
use crate::encoder::stateless::BitstreamPromise;
use crate::encoder::stateless::FrameMetadata;
use crate::encoder::stateless::Predictor;
use crate::encoder::stateless::StatelessBackendResult;
use crate::encoder::stateless::StatelessCodec;
use crate::encoder::stateless::StatelessEncoderBackendImport;
use crate::encoder::stateless::StatelessEncoderExecute;
use crate::encoder::stateless::StatelessVideoEncoderBackend;
use crate::encoder::EncodeResult;
use crate::encoder::PredictionStructure;
use crate::encoder::Tunings;
use crate::BlockingMode;

mod predictor;

#[cfg(feature = "vaapi")]
pub mod vaapi;

#[cfg(test)]
#[path = "h265/tests.rs"]
mod tests;

/// Decoded picture buffer entry metadata. Unlike H.264 there is no `frame_num`:
/// HEVC orders references purely by picture order count.
#[derive(Clone, Debug)]
pub struct DpbEntryMeta {
    /// Picture order count (the full value, not the LSB).
    poc: i32,
    /// The NAL unit type the reconstructed picture was coded as.
    #[allow(dead_code)]
    nalu_type: NaluType,
}

/// Frame structure used in the backend representing currently encoded frame or references used
/// for its encoding.
pub struct DpbEntry<R> {
    /// Reconstructed picture
    recon_pic: R,
    /// Decoded picture buffer entry metadata
    meta: DpbEntryMeta,
}

/// Stateless HEVC encoder backend input.
pub struct BackendRequest<P, R> {
    /// The active video parameter set.
    vps: Rc<Vps>,
    /// The active sequence parameter set.
    sps: Rc<Sps>,
    /// The active picture parameter set.
    pps: Rc<Pps>,
    /// The slice segment header for the (single) slice. Carries the in-header
    /// short-term reference picture set (`header.short_term_ref_pic_set`, the
    /// trivial LowDelay form: one negative reference, `delta_poc_s0_minus1 == 0`).
    header: SliceHeader,
    /// The NAL unit type of the coded slice (`IdrWRadl` / `TrailR`).
    nalu_type: NaluType,

    /// Input frame to be encoded
    input: P,

    /// Input frame metadata
    input_meta: FrameMetadata,

    /// DPB entry metadata
    dpb_meta: DpbEntryMeta,

    /// Reference list 0 (LowDelay P uses list 0 only).
    ref_list_0: Vec<Rc<DpbEntry<R>>>,

    /// Period between intra frames
    intra_period: u32,

    /// Period between intra frame and P frame
    ip_period: u32,

    /// Number of coding tree units to be encoded in the slice.
    num_ctu_in_slice: u32,

    /// True whenever the result is IDR (drives which application-packed headers
    /// the VA-API backend emits: VPS + SPS on IDR, PPS every frame).
    is_idr: bool,

    /// [`Tunings`] for the frame
    tunings: Tunings,

    /// Container for the request output. The [`StatelessH265EncoderBackend`] impl
    /// moves it and appends the coded bitstream to it, avoiding a copy. For HEVC
    /// this is empty on entry — the parameter sets are supplied to the driver via
    /// application-packed headers (or self-generated), so they land directly in
    /// the driver's coded buffer rather than being prepended here.
    coded_output: Vec<u8>,
}

/// Wrapper type for [`BackendPromise<Output = R>`], with additional metadata.
pub struct ReferencePromise<P>
where
    P: BackendPromise,
{
    /// Slice data and reconstructed surface promise
    recon: P,

    /// [`DpbEntryMeta`] of reconstructed surface
    dpb_meta: DpbEntryMeta,
}

impl<P> BackendPromise for ReferencePromise<P>
where
    P: BackendPromise,
{
    type Output = DpbEntry<P::Output>;

    fn is_ready(&self) -> bool {
        self.recon.is_ready()
    }

    fn sync(self) -> StatelessBackendResult<Self::Output> {
        let recon_pic = self.recon.sync()?;

        log::trace!("synced recon picture poc={}", self.dpb_meta.poc);

        Ok(DpbEntry { recon_pic, meta: self.dpb_meta })
    }
}

impl<Backend> StatelessCodec<Backend> for H265
where
    Backend: StatelessVideoEncoderBackend<H265>,
{
    type Reference = DpbEntry<Backend::Reconstructed>;

    type Request = BackendRequest<Backend::Picture, Backend::Reconstructed>;

    type CodedPromise = BitstreamPromise<Backend::CodedPromise>;

    type ReferencePromise = ReferencePromise<Backend::ReconPromise>;
}

/// Trait for stateless encoder backend for HEVC.
pub trait StatelessH265EncoderBackend: StatelessVideoEncoderBackend<H265> {
    /// Submit a [`BackendRequest`] to the backend. This operation returns both a
    /// [`StatelessVideoEncoderBackend::CodedPromise`] and a
    /// [`StatelessVideoEncoderBackend::ReconPromise`] with resulting slice data.
    fn encode_slice(
        &mut self,
        request: BackendRequest<Self::Picture, Self::Reconstructed>,
    ) -> StatelessBackendResult<(Self::ReconPromise, Self::CodedPromise)>;
}

pub type StatelessEncoder<Handle, Backend> =
    crate::encoder::stateless::StatelessEncoder<H265, Handle, Backend>;

impl<Handle, Backend> StatelessEncoderExecute<H265, Handle, Backend>
    for StatelessEncoder<Handle, Backend>
where
    Backend: StatelessH265EncoderBackend,
{
    fn execute(
        &mut self,
        request: BackendRequest<Backend::Picture, Backend::Reconstructed>,
    ) -> EncodeResult<()> {
        let meta = request.input_meta.clone();
        let dpb_meta = request.dpb_meta.clone();

        // The [`BackendRequest`] has a frame from predictor. Decreasing internal counter.
        self.predictor_frame_count -= 1;

        log::trace!("submitting new request");
        let (recon, bitstream) = self.backend.encode_slice(request)?;

        // Wrap promise from backend with headers and metadata
        let slice_promise = BitstreamPromise { bitstream, meta };

        self.output_queue.add_promise(slice_promise);

        let ref_promise = ReferencePromise { recon, dpb_meta };

        self.recon_queue.add_promise(ref_promise);

        Ok(())
    }
}

impl<Handle, Backend> StatelessEncoder<Handle, Backend>
where
    Backend: StatelessH265EncoderBackend,
    Backend: StatelessEncoderBackendImport<Handle, Backend::Picture>,
{
    fn new_h265(backend: Backend, config: EncoderConfig, mode: BlockingMode) -> EncodeResult<Self> {
        let predictor: Box<dyn Predictor<_, _, _>> = match config.pred_structure {
            PredictionStructure::LowDelay { limit } => Box::new(LowDelayH265::new(config, limit)),
        };

        Self::new(backend, mode, predictor)
    }
}
