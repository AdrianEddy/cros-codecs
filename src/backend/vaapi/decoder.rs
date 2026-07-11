// Copyright 2023 The ChromiumOS Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

use std::cell::RefCell;
use std::marker::PhantomData;
use std::rc::Rc;
use std::sync::Arc;

use anyhow::anyhow;
use anyhow::Context as AnyhowContext;
use libva::{
    Buffer, Context, Display, Picture, PictureEnd, PictureNew, PictureSync, Surface,
    SurfaceMemoryDescriptor, VaError,
};

use crate::decoder::stateless::StatelessBackendResult;
use crate::decoder::stateless::StatelessCodec;
use crate::decoder::stateless::StatelessDecoderBackend;
use crate::decoder::stateless::StatelessDecoderBackendPicture;
use crate::decoder::DecodedHandle as DecodedHandleTrait;
use crate::decoder::StreamInfo;
use crate::video_frame::VideoFrame;
use crate::DecodedFormat;
use crate::Rect;
use crate::Resolution;

/// A decoded frame handle.
pub(crate) type DecodedHandle<V> = Rc<RefCell<VaapiDecodedHandle<V>>>;

/// Gets the VASurfaceID for the given `picture`.
pub(crate) fn va_surface_id<V: VideoFrame>(
    handle: &Option<DecodedHandle<V>>,
) -> libva::VASurfaceID {
    match handle {
        None => libva::VA_INVALID_SURFACE,
        Some(handle) => handle.borrow().surface().id(),
    }
}

impl<V: VideoFrame> DecodedHandleTrait for DecodedHandle<V> {
    type Frame = V;

    fn video_frame(&self) -> Arc<Self::Frame> {
        self.borrow().backing_frame.clone()
    }

    /// The grain-applied display frame when this handle carries one (AV1 film
    /// grain, [`VaapiDisplayTarget`]), otherwise the reconstruct — which is what
    /// [`Self::video_frame`] (and thus the DPB references) return. Overrides the
    /// trait default so output ≠ DPB reference for grain frames.
    fn display_frame(&self) -> Arc<Self::Frame> {
        let handle = self.borrow();
        match &handle.display {
            Some(display) => display.frame.clone(),
            None => handle.backing_frame.clone(),
        }
    }

    fn coded_resolution(&self) -> Resolution {
        self.borrow().surface().size().into()
    }

    fn display_resolution(&self) -> Resolution {
        self.borrow().display_resolution
    }

    fn timestamp(&self) -> u64 {
        self.borrow().timestamp()
    }

    fn is_ready(&self) -> bool {
        self.borrow().state.is_ready().unwrap_or(true)
    }

    fn sync(&self) -> anyhow::Result<()> {
        self.borrow_mut().sync().context("while syncing picture")?;

        Ok(())
    }
}

/// A trait for providing the basic information needed to setup libva for decoding.
pub(crate) trait VaStreamInfo {
    /// Returns the VA profile of the stream.
    fn va_profile(&self) -> anyhow::Result<i32>;
    /// Returns the RT format of the stream.
    fn rt_format(&self) -> anyhow::Result<u32>;
    /// Returns the minimum number of surfaces required to decode the stream.
    fn min_num_surfaces(&self) -> usize;
    /// Returns the coded size of the surfaces required to decode the stream.
    fn coded_size(&self) -> Resolution;
    /// Returns the visible rectangle within the coded size for the stream.
    fn visible_rect(&self) -> Rect;
}

/// Rendering state of a VA picture.
enum PictureState<M: SurfaceMemoryDescriptor> {
    Ready(Picture<PictureSync, Surface<M>>),
    Pending(Picture<PictureEnd, Surface<M>>),
    // Only set in sync when we take ownership of the VA picture.
    Invalid,
}

impl<M: SurfaceMemoryDescriptor> PictureState<M> {
    /// Make sure that all pending operations on the picture have completed.
    fn sync(&mut self) -> Result<(), VaError> {
        let res;

        (*self, res) = match std::mem::replace(self, PictureState::Invalid) {
            state @ PictureState::Ready(_) => (state, Ok(())),
            PictureState::Pending(picture) => match picture.sync() {
                Ok(picture) => (PictureState::Ready(picture), Ok(())),
                Err((e, picture)) => (PictureState::Pending(picture), Err(e)),
            },
            PictureState::Invalid => unreachable!(),
        };

        res
    }

    fn surface(&self) -> &Surface<M> {
        match self {
            PictureState::Ready(picture) => picture.surface(),
            PictureState::Pending(picture) => picture.surface(),
            PictureState::Invalid => unreachable!(),
        }
    }

    fn timestamp(&self) -> u64 {
        match self {
            PictureState::Ready(picture) => picture.timestamp(),
            PictureState::Pending(picture) => picture.timestamp(),
            PictureState::Invalid => unreachable!(),
        }
    }

    fn is_ready(&self) -> Result<bool, VaError> {
        match self {
            PictureState::Ready(_) => Ok(true),
            PictureState::Pending(picture) => picture
                .surface()
                .query_status()
                .map(|s| s == libva::VASurfaceStatus::VASurfaceReady),
            PictureState::Invalid => unreachable!(),
        }
    }

    fn new_from_same_surface(&self, timestamp: u64) -> Picture<PictureNew, Surface<M>> {
        match &self {
            PictureState::Ready(picture) => Picture::new_from_same_surface(timestamp, picture),
            PictureState::Pending(picture) => Picture::new_from_same_surface(timestamp, picture),
            PictureState::Invalid => unreachable!(),
        }
    }
}

/// The grain-applied display target of an AV1 film-grain frame.
///
/// AV1 film grain is applied outside the coding loop, so the frame kept as a DPB
/// reference must stay grain-free while the presented frame carries the grain.
/// When a picture applies grain, the decode renders the grain-free reconstruct
/// into the [`VaapiPicture`]/[`VaapiDecodedHandle`] `backing_frame` surface (the
/// DPB reference) and the driver writes the grain-applied output into this
/// separate surface (`current_display_picture` in the AV1 picture parameter).
/// Both come from the same caller pool (a second `alloc_cb`). `frame` is what
/// [`DecodedHandle::display_frame`](DecodedHandleTrait::display_frame) returns;
/// `surface` is synced alongside the reconstruct. `None` for every non-grain
/// frame (all other codecs, and grain-free AV1 frames — output == reconstruct).
struct VaapiDisplayTarget<V: VideoFrame> {
    frame: Arc<V>,
    surface: Surface<<V as VideoFrame>::MemDescriptor>,
}

/// VA-API backend handle.
///
/// This includes the VA picture which can be pending rendering or complete, as well as useful
/// meta-information.
pub struct VaapiDecodedHandle<V: VideoFrame> {
    backing_frame: Arc<V>,
    state: PictureState<<V as VideoFrame>::MemDescriptor>,
    /// Actual resolution of the visible rectangle in the decoded buffer.
    display_resolution: Resolution,
    /// The grain-applied display target when this handle applies AV1 film grain;
    /// `None` when output == reconstruct (the common case).
    display: Option<VaapiDisplayTarget<V>>,
}

impl<V: VideoFrame> VaapiDecodedHandle<V> {
    /// Creates a new pending handle on `surface_id`.
    fn new(picture: VaapiPicture<V>, display_resolution: Resolution) -> anyhow::Result<Self> {
        let backing_frame = picture.backing_frame;
        let display = picture.display;
        let picture = picture.picture.begin()?.render()?.end()?;
        Ok(Self {
            backing_frame: backing_frame,
            state: PictureState::Pending(picture),
            display_resolution: display_resolution,
            display,
        })
    }

    fn sync(&mut self) -> Result<(), VaError> {
        self.state.sync()?;
        // The grain-applied display surface is written by the same decode
        // operation but is a distinct surface, so it must be synced too before
        // the display frame is read/exported.
        if let Some(display) = &self.display {
            display.surface.sync()?;
        }
        Ok(())
    }

    /// Creates a new picture from the surface backing the current one. Useful for interlaced
    /// decoding. TODO: Do we need this for other purposes? We don't intend to support interlaced.
    pub(crate) fn new_picture_from_same_surface(&self, timestamp: u64) -> VaapiPicture<V> {
        VaapiPicture {
            picture: self.state.new_from_same_surface(timestamp),
            backing_frame: self.backing_frame.clone(),
            display: None,
        }
    }

    pub(crate) fn surface(&self) -> &Surface<<V as VideoFrame>::MemDescriptor> {
        self.state.surface()
    }

    /// Returns the timestamp of this handle.
    fn timestamp(&self) -> u64 {
        self.state.timestamp()
    }
}

pub struct VaapiBackend<V: VideoFrame> {
    pub display: Rc<Display>,
    pub context: Rc<Context>,
    stream_info: StreamInfo,
    // TODO: We should try to support context reuse
    _supports_context_reuse: bool,
    _phantom_data: PhantomData<V>,
}

/// Maps a stream `RT_FORMAT` to the [`DecodedFormat`] of the native
/// surfaces we export for zero-copy DMA output. Only the three formats that have
/// output wiring are supported:
/// - `YUV420`    → [`DecodedFormat::NV12`] (8-bit 4:2:0)
/// - `YUV420_10` → [`DecodedFormat::P010`] (10-bit 4:2:0, 2-plane semi-planar,
///   MSB-justified)
/// - `YUV422_10` → [`DecodedFormat::Y210`] (10-bit 4:2:2, single packed plane,
///   MSB-justified)
///
/// The reported format describes the actual surface layout — the planar,
/// LSB-justified [`DecodedFormat::I010`]/[`DecodedFormat::I210`] would be a lie
/// for these surfaces and mislead `StreamInfo` consumers and CPU readback.
///
/// 12-bit (P012/Y212) and 4:4:4 (Y410/Y412) have no output wiring yet and are
/// rejected with an explicit error, so profiles such as HEVC Main12/Main444
/// still fail cleanly here instead of
/// decoding into a wrong-layout surface.
fn output_decoded_format(rt_format: u32) -> anyhow::Result<DecodedFormat> {
    match rt_format {
        libva::VA_RT_FORMAT_YUV420 => Ok(DecodedFormat::NV12),
        libva::VA_RT_FORMAT_YUV420_10 => Ok(DecodedFormat::P010),
        libva::VA_RT_FORMAT_YUV422_10 => Ok(DecodedFormat::Y210),
        other => Err(anyhow!(
            "Unsupported decode output RT format {:#x}: only 8-bit 4:2:0 (NV12), \
             10-bit 4:2:0 (P010) and 10-bit 4:2:2 (Y210) are wired for output",
            other
        )),
    }
}

impl<V: VideoFrame> VaapiBackend<V> {
    // The initial config/context creation returns a
    // `Result` instead of `.expect()`-panicking, so a driver that cannot create
    // the bootstrap H.264-Main VLD config/context surfaces a `VaError` the caller
    // classifies, rather than aborting the process.
    pub(crate) fn new(
        display: Rc<libva::Display>,
        supports_context_reuse: bool,
    ) -> Result<Self, VaError> {
        let init_stream_info = StreamInfo {
            format: DecodedFormat::NV12,
            coded_resolution: Resolution::from((16, 16)),
            display_resolution: Resolution::from((16, 16)),
            min_num_frames: 1,
        };
        let config = display.create_config(
            vec![libva::VAConfigAttrib {
                type_: libva::VAConfigAttribType::VAConfigAttribRTFormat,
                value: libva::VA_RT_FORMAT_YUV420,
            }],
            libva::VAProfile::VAProfileH264Main,
            libva::VAEntrypoint::VAEntrypointVLD,
        )?;
        let context = display.create_context::<<V as VideoFrame>::MemDescriptor>(
            &config,
            init_stream_info.coded_resolution.width,
            init_stream_info.coded_resolution.height,
            None,
            true,
        )?;
        Ok(Self {
            display: display,
            context: context,
            _supports_context_reuse: supports_context_reuse,
            stream_info: init_stream_info,
            _phantom_data: Default::default(),
        })
    }

    pub(crate) fn new_sequence<StreamData>(
        &mut self,
        stream_params: &StreamData,
    ) -> StatelessBackendResult<()>
    where
        for<'a> &'a StreamData: VaStreamInfo,
    {
        self.stream_info.display_resolution = Resolution::from(stream_params.visible_rect());
        self.stream_info.coded_resolution = stream_params.coded_size().clone();
        self.stream_info.min_num_frames = stream_params.min_num_surfaces();

        // Derive the RT format and the output `DecodedFormat` from the
        // stream's bit depth and chroma subsampling instead of hardcoding 8-bit
        // 4:2:0. Only the three formats with zero-copy DMA output wiring are
        // accepted (see `output_decoded_format`); any other (12-bit, 4:4:4, …)
        // is an explicit error rather than a silent mis-decode into a
        // wrong-layout surface.
        let rt_format = stream_params
            .rt_format()
            .map_err(|_| anyhow!("Could not get RT format from stream!"))?;
        self.stream_info.format = output_decoded_format(rt_format)?;

        // TODO: Handle context re-use
        let config = self
            .display
            .create_config(
                vec![libva::VAConfigAttrib {
                    type_: libva::VAConfigAttribType::VAConfigAttribRTFormat,
                    value: rt_format,
                }],
                stream_params.va_profile().map_err(|_| anyhow!("Could not get VAProfile!"))?,
                libva::VAEntrypoint::VAEntrypointVLD,
            )
            .map_err(|_| anyhow!("Could not create VAConfig!"))?;
        let context = self
            .display
            .create_context::<<V as VideoFrame>::MemDescriptor>(
                &config,
                self.stream_info.coded_resolution.width,
                self.stream_info.coded_resolution.height,
                None,
                true,
            )
            .map_err(|_| anyhow!("Could not create VAContext!"))?;
        self.context = context;

        Ok(())
    }

    pub(crate) fn process_picture<Codec: StatelessCodec>(
        &mut self,
        picture: VaapiPicture<V>,
    ) -> StatelessBackendResult<<Self as StatelessDecoderBackend>::Handle>
    where
        Self: StatelessDecoderBackendPicture<Codec>,
        for<'a> &'a Codec::FormatInfo: VaStreamInfo,
    {
        Ok(Rc::new(RefCell::new(VaapiDecodedHandle::new(
            picture,
            self.stream_info.display_resolution.clone(),
        )?)))
    }
}

/// Shortcut for pictures used for the VAAPI backend.
pub struct VaapiPicture<V: VideoFrame> {
    picture: Picture<PictureNew, Surface<V::MemDescriptor>>,
    backing_frame: Arc<V>,
    /// The grain-applied display target for AV1 film grain; `None` when
    /// output == reconstruct. Threaded onto the [`VaapiDecodedHandle`] at submit.
    display: Option<VaapiDisplayTarget<V>>,
}

impl<V: VideoFrame> VaapiPicture<V> {
    // The PRIME_2 decode-target re-import
    // (`to_native_handle`) returns a `Result` instead of `.expect()`-panicking,
    // so a driver that rejects the imported dmabuf (e.g. an unsupported modifier)
    // surfaces an error the caller classifies rather than aborting the process.
    pub fn new(timestamp: u64, context: Rc<Context>, backing_frame: V) -> anyhow::Result<Self> {
        let display = context.display();
        let surface = backing_frame.to_native_handle(display).map_err(|e| anyhow!(e))?.into();
        Ok(Self {
            backing_frame: Arc::new(backing_frame),
            picture: Picture::new(timestamp, context, surface),
            display: None,
        })
    }

    /// Creates a picture with a distinct grain-applied display target for AV1
    /// film grain. The decode renders the grain-free reconstruct into
    /// `backing_frame`'s surface (the DPB reference / `current_frame`), and the
    /// driver writes the grain-applied output into `display_frame`'s surface
    /// (`current_display_picture`). Both frames come from the same caller pool.
    pub fn new_with_display(
        timestamp: u64,
        context: Rc<Context>,
        backing_frame: V,
        display_frame: V,
    ) -> anyhow::Result<Self> {
        let display = context.display();
        let surface = backing_frame.to_native_handle(display).map_err(|e| anyhow!(e))?.into();
        let display_surface = display_frame.to_native_handle(display).map_err(|e| anyhow!(e))?.into();
        Ok(Self {
            backing_frame: Arc::new(backing_frame),
            picture: Picture::new(timestamp, context, surface),
            display: Some(VaapiDisplayTarget {
                frame: Arc::new(display_frame),
                surface: display_surface,
            }),
        })
    }

    pub fn surface(&self) -> &Surface<V::MemDescriptor> {
        self.picture.surface()
    }

    /// The VA surface id of the grain-applied display target, or
    /// [`libva::VA_INVALID_SURFACE`] when there is none — the value the AV1
    /// picture parameter's `current_display_picture` field expects (grain-free
    /// frames leave it invalid so the driver applies no grain).
    pub fn display_surface_id(&self) -> libva::VASurfaceID {
        self.display.as_ref().map_or(libva::VA_INVALID_SURFACE, |d| d.surface.id())
    }

    pub fn add_buffer(&mut self, buffer: Buffer) {
        self.picture.add_buffer(buffer)
    }
}

impl<V: VideoFrame> StatelessDecoderBackend for VaapiBackend<V> {
    type Handle = DecodedHandle<V>;

    fn stream_info(&self) -> Option<&StreamInfo> {
        Some(&self.stream_info)
    }

    fn reset_backend(&mut self) -> anyhow::Result<()> {
        //TODO(bchoobineh): Implement VAAPI DRC
        Ok(())
    }
}
