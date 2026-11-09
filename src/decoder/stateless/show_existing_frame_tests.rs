// Copyright 2026 The ChromiumOS Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

//! A picture an AV1 or VP9 `show_existing_frame` shows comes out under the
//! timestamp of the unit that shows it. A VP9 superframe short of output buffers
//! asks only for its coded frames' pictures, whatever show-existing frames it holds.
//!
//! The decoders run on real streams over a host-only backend that decodes nothing:
//! it numbers the pictures it is asked to decode, records the timestamp each was
//! decoded under, and hands out handles that say which picture they show. So every
//! output can be checked for its timestamp (the unit's that shows it, one output per
//! unit) and for its picture (a show-existing output shows the picture the named
//! reference slot holds, sharing it with the handle that decoded it).
//!
//! Streams:
//! - `vp90-2-10-show-existing-frame.vp9.ivf` and `...-frame2.vp9.ivf`, libvpx's
//!   conformance vectors: hidden pictures shown later, shown pictures shown again,
//!   and a show-existing frame inside a superframe.
//! - `show-existing-frame-10bit.ivf`: the 30 temporal units of oxivideo's
//!   `av1_10bit.mp4` (an encoder's hidden alternate references, each shown once by a
//!   later unit: 14 show-existing frame headers).
//! - The first 12 units of `test-25fps.av1.ivf` with a unit inserted after unit 2
//!   that shows unit 2's picture again (the shown inter picture in reference slot 1).
//! - Superframes of the first VP9 vector's key frame and one-byte show-existing
//!   frames, built here.

use std::cell::Cell;
use std::rc::Rc;
use std::sync::Arc;

use crate::bitstream_utils::IvfIterator;
use crate::codec::av1::parser::FrameHeaderObu;
use crate::codec::av1::parser::StreamInfo as Av1StreamInfo;
use crate::codec::av1::parser::TileGroupObu;
use crate::codec::av1::parser::NUM_REF_FRAMES as AV1_NUM_REF_FRAMES;
use crate::codec::vp9::parser::Header as Vp9Header;
use crate::codec::vp9::parser::Segmentation;
use crate::codec::vp9::parser::MAX_SEGMENTS;
use crate::codec::vp9::parser::NUM_REF_FRAMES as VP9_NUM_REF_FRAMES;
use crate::decoder::stateless::av1::Av1;
use crate::decoder::stateless::av1::StatelessAV1DecoderBackend;
use crate::decoder::stateless::vp9::StatelessVp9DecoderBackend;
use crate::decoder::stateless::vp9::Vp9;
use crate::decoder::stateless::DecodeError;
use crate::decoder::stateless::NewPictureError;
use crate::decoder::stateless::NewPictureResult;
use crate::decoder::stateless::StatelessBackendResult;
use crate::decoder::stateless::StatelessCodec;
use crate::decoder::stateless::StatelessDecoder;
use crate::decoder::stateless::StatelessDecoderBackend;
use crate::decoder::stateless::StatelessDecoderBackendPicture;
use crate::decoder::stateless::StatelessVideoDecoder;
use crate::decoder::BlockingMode;
use crate::decoder::DecodedHandle;
use crate::decoder::DecoderEvent;
use crate::decoder::StreamInfo;
use crate::video_frame::ReadMapping;
use crate::video_frame::VideoFrame;
use crate::video_frame::WriteMapping;
use crate::DecodedFormat;
use crate::Fourcc;
use crate::Resolution;

/// A frame with no memory: the backend below decodes nothing into it.
#[derive(Debug)]
struct HostFrame;

impl VideoFrame for HostFrame {
    #[cfg(feature = "v4l2")]
    type NativeHandle = v4l2r::memory::MmapHandle;

    #[cfg(feature = "vaapi")]
    type MemDescriptor = ();
    #[cfg(feature = "vaapi")]
    type NativeHandle = libva::Surface<()>;

    fn fourcc(&self) -> Fourcc {
        Fourcc::from(b"NV12")
    }

    fn resolution(&self) -> Resolution {
        Resolution { width: 0, height: 0 }
    }

    fn get_plane_size(&self) -> Vec<usize> {
        unreachable!("the host backend maps no frame")
    }

    fn get_plane_pitch(&self) -> Vec<usize> {
        unreachable!("the host backend maps no frame")
    }

    fn map<'a>(&'a self) -> Result<Box<dyn ReadMapping<'a> + 'a>, String> {
        unreachable!("the host backend maps no frame")
    }

    fn map_mut<'a>(&'a mut self) -> Result<Box<dyn WriteMapping<'a> + 'a>, String> {
        unreachable!("the host backend maps no frame")
    }

    #[cfg(feature = "v4l2")]
    fn fill_v4l2_plane(&self, _index: usize, _plane: &mut v4l2r::bindings::v4l2_plane) {
        unreachable!("the host backend queues no V4L2 buffer")
    }

    #[cfg(feature = "v4l2")]
    fn process_dqbuf(
        &mut self,
        _device: Arc<v4l2r::device::Device>,
        _format: &v4l2r::Format,
        _buf: &v4l2r::ioctl::V4l2Buffer,
    ) {
        unreachable!("the host backend queues no V4L2 buffer")
    }

    #[cfg(feature = "vaapi")]
    fn to_native_handle(
        &self,
        _display: &Rc<libva::Display>,
    ) -> Result<Self::NativeHandle, String> {
        unreachable!("the host backend creates no VA surface")
    }
}

/// A picture the host backend was asked to decode: its decode order and the
/// timestamp it was decoded under.
#[derive(Debug)]
struct HostPicture {
    index: usize,
    decoded_at: u64,
    frame: Arc<HostFrame>,
}

#[derive(Clone, Debug)]
struct HostHandle {
    picture: Rc<HostPicture>,
    timestamp: u64,
}

impl DecodedHandle for HostHandle {
    type Frame = HostFrame;

    fn video_frame(&self) -> Arc<HostFrame> {
        Arc::clone(&self.picture.frame)
    }

    fn timestamp(&self) -> u64 {
        self.timestamp
    }

    fn coded_resolution(&self) -> Resolution {
        Resolution { width: 0, height: 0 }
    }

    fn display_resolution(&self) -> Resolution {
        Resolution { width: 0, height: 0 }
    }

    fn is_ready(&self) -> bool {
        true
    }

    fn sync(&self) -> anyhow::Result<()> {
        Ok(())
    }
}

/// The picture being decoded: its timestamp and the frame `alloc_cb` gave it.
struct PendingPicture {
    timestamp: u64,
    frame: HostFrame,
}

#[derive(Default)]
struct HostBackend {
    stream_info: Option<StreamInfo>,
    decoded: Cell<usize>,
}

impl HostBackend {
    fn negotiate(&mut self, width: u32, height: u32) {
        let resolution = Resolution { width, height };
        self.stream_info = Some(StreamInfo {
            format: DecodedFormat::NV12,
            coded_resolution: resolution,
            display_resolution: resolution,
            min_num_frames: VP9_NUM_REF_FRAMES.max(AV1_NUM_REF_FRAMES),
        });
    }

    fn pending_picture(
        timestamp: u64,
        alloc_cb: &mut dyn FnMut() -> Option<HostFrame>,
    ) -> NewPictureResult<PendingPicture> {
        let frame = alloc_cb().ok_or(NewPictureError::OutOfOutputBuffers)?;
        Ok(PendingPicture { timestamp, frame })
    }

    fn submit(&self, picture: PendingPicture) -> HostHandle {
        let index = self.decoded.get();
        self.decoded.set(index + 1);
        HostHandle {
            picture: Rc::new(HostPicture {
                index,
                decoded_at: picture.timestamp,
                frame: Arc::new(picture.frame),
            }),
            timestamp: picture.timestamp,
        }
    }
}

impl StatelessDecoderBackend for HostBackend {
    type Handle = HostHandle;

    fn stream_info(&self) -> Option<&StreamInfo> {
        self.stream_info.as_ref()
    }

    fn reset_backend(&mut self) -> anyhow::Result<()> {
        Ok(())
    }

    fn duplicate_handle(&self, handle: &HostHandle, timestamp: u64) -> HostHandle {
        HostHandle { picture: Rc::clone(&handle.picture), timestamp }
    }
}

impl StatelessDecoderBackendPicture<Vp9> for HostBackend {
    type Picture = PendingPicture;
}

impl StatelessVp9DecoderBackend for HostBackend {
    fn new_sequence(&mut self, header: &Vp9Header) -> StatelessBackendResult<()> {
        self.negotiate(header.width, header.height);
        Ok(())
    }

    fn new_picture(
        &mut self,
        timestamp: u64,
        alloc_cb: &mut dyn FnMut() -> Option<HostFrame>,
    ) -> NewPictureResult<PendingPicture> {
        Self::pending_picture(timestamp, alloc_cb)
    }

    fn submit_picture(
        &mut self,
        picture: PendingPicture,
        _hdr: &Vp9Header,
        _reference_frames: &[Option<HostHandle>; VP9_NUM_REF_FRAMES],
        _bitstream: &[u8],
        _segmentation: &[Segmentation; MAX_SEGMENTS],
    ) -> StatelessBackendResult<HostHandle> {
        Ok(self.submit(picture))
    }
}

impl StatelessDecoderBackendPicture<Av1> for HostBackend {
    type Picture = PendingPicture;
}

impl StatelessAV1DecoderBackend for HostBackend {
    fn change_stream_info(&mut self, stream_info: &Av1StreamInfo) -> StatelessBackendResult<()> {
        self.negotiate(stream_info.render_width, stream_info.render_height);
        Ok(())
    }

    fn new_picture(
        &mut self,
        _hdr: &FrameHeaderObu,
        timestamp: u64,
        alloc_cb: &mut dyn FnMut() -> Option<HostFrame>,
    ) -> NewPictureResult<PendingPicture> {
        Self::pending_picture(timestamp, alloc_cb)
    }

    fn begin_picture(
        &mut self,
        _picture: &mut PendingPicture,
        _stream_info: &Av1StreamInfo,
        _hdr: &FrameHeaderObu,
        _reference_frames: &[Option<HostHandle>; AV1_NUM_REF_FRAMES],
    ) -> StatelessBackendResult<()> {
        Ok(())
    }

    fn decode_tile_group(
        &mut self,
        _picture: &mut PendingPicture,
        _tile_group: TileGroupObu,
    ) -> StatelessBackendResult<()> {
        Ok(())
    }

    fn submit_picture(&mut self, picture: PendingPicture) -> StatelessBackendResult<HostHandle> {
        Ok(self.submit(picture))
    }
}

/// Decode `units` in order, unit `i` under timestamp `i`, and return every output
/// in output order.
fn decode_units<C>(units: &[Vec<u8>]) -> Vec<HostHandle>
where
    C: StatelessCodec,
    C::DecoderState<HostHandle, PendingPicture>: Default,
    HostBackend: StatelessDecoderBackendPicture<C, Picture = PendingPicture>,
    StatelessDecoder<C, HostBackend>: StatelessVideoDecoder<Handle = HostHandle>,
{
    let mut decoder =
        StatelessDecoder::<C, HostBackend>::new(HostBackend::default(), BlockingMode::Blocking)
            .expect("a decoder on the host backend");
    let mut outputs = Vec::new();
    let mut drain = |decoder: &mut StatelessDecoder<C, HostBackend>| {
        while let Some(event) = decoder.next_event() {
            match event {
                DecoderEvent::FrameReady(handle) => outputs.push(handle),
                DecoderEvent::FormatChanged => (),
            }
        }
    };
    for (timestamp, unit) in units.iter().enumerate() {
        let mut consumed = 0;
        while consumed < unit.len() {
            match decoder.decode(timestamp as u64, &unit[consumed..], &mut || Some(HostFrame)) {
                Ok(n) => consumed += n,
                Err(DecodeError::CheckEvents) => (),
                Err(e) => panic!("unit {timestamp}: {e}"),
            }
            drain(&mut decoder);
        }
    }
    decoder.flush().expect("flush");
    drain(&mut decoder);
    outputs
}

fn ivf_units(stream: &[u8]) -> Vec<Vec<u8>> {
    IvfIterator::new(stream).map(<[u8]>::to_vec).collect()
}

/// One output per unit, each under its unit's timestamp, so the timestamps rise.
fn assert_one_output_per_unit(outputs: &[HostHandle], units: usize) {
    let timestamps: Vec<u64> = outputs.iter().map(|h| h.timestamp).collect();
    assert_eq!(timestamps, (0..units as u64).collect::<Vec<_>>(), "one output per unit, in order");
}

/// The outputs that show the picture an earlier output showed: `(earlier, later)`.
fn shown_again(outputs: &[HostHandle]) -> Vec<(usize, usize)> {
    let mut pairs = Vec::new();
    for (later, handle) in outputs.iter().enumerate() {
        if let Some(earlier) =
            outputs[..later].iter().rposition(|o| Rc::ptr_eq(&o.picture, &handle.picture))
        {
            pairs.push((earlier, later));
        }
    }
    pairs
}

/// libvpx's first show-existing vector: a superframe of two hidden pictures and a
/// show-existing frame of the first (unit 1), a lone show-existing frame of the
/// second (unit 2), and shown pictures shown again by later units (units 3 → 5,
/// 6 → 7 inside a superframe, 9 → 11, 10 → 12).
#[test]
fn vp9_show_existing_frames_come_out_under_the_showing_unit() {
    let units = ivf_units(include_bytes!(
        "../../codec/vp9/test_data/vp90-2-10-show-existing-frame.vp9.ivf"
    ));
    assert_eq!(units.len(), 13);
    let outputs = decode_units::<Vp9>(&units);
    assert_one_output_per_unit(&outputs, units.len());
    assert_eq!(shown_again(&outputs), [(3, 5), (6, 7), (9, 11), (10, 12)]);
    // Units 1 and 2 show the two hidden pictures unit 1 decoded.
    for unit in [1, 2] {
        assert_eq!(outputs[unit].picture.decoded_at, 1, "unit {unit} shows a picture of unit 1");
    }
    assert_ne!(outputs[1].picture.index, outputs[2].picture.index);
    // Unit 4 shows the hidden picture of the superframe in unit 3.
    assert_eq!(outputs[4].picture.decoded_at, 3);
}

/// libvpx's second show-existing vector: units 8 to 15 each show again a picture
/// one of units 3 to 7 showed, three of them twice.
#[test]
fn vp9_pictures_shown_again_come_out_under_each_showing_unit() {
    let units = ivf_units(include_bytes!(
        "../../codec/vp9/test_data/vp90-2-10-show-existing-frame2.vp9.ivf"
    ));
    assert_eq!(units.len(), 16);
    let outputs = decode_units::<Vp9>(&units);
    assert_one_output_per_unit(&outputs, units.len());
    assert_eq!(
        shown_again(&outputs),
        [(3, 8), (4, 9), (5, 10), (6, 11), (7, 12), (8, 13), (9, 14), (10, 15)]
    );
    // Units 3 to 7 decoded the pictures they show.
    for (unit, handle) in outputs.iter().enumerate().take(8).skip(3) {
        assert_eq!(handle.picture.decoded_at, unit as u64);
    }
}

/// A VP9 frame header that shows reference slot 0 again: frame_marker `0b10`,
/// profile 0, show_existing_frame 1, frame_to_show_map_idx 0.
const VP9_SHOW_SLOT_0: [u8; 1] = [0b1000_1000];

/// A VP9 superframe of `frames` (Annex B): their bytes, then an index of their sizes
/// between two marker bytes `0b110_mm_nnn`, here with 4-byte sizes (`mm` 3) and
/// `nnn + 1` frames.
fn vp9_superframe(frames: &[&[u8]]) -> Vec<u8> {
    let marker = 0b1101_1000 | (frames.len() - 1) as u8;
    let mut superframe = frames.concat();
    superframe.push(marker);
    for frame in frames {
        superframe.extend_from_slice(&(frame.len() as u32).to_le_bytes());
    }
    superframe.push(marker);
    superframe
}

/// Decode `unit` under `timestamp` with `buffers` output buffers to give, past a
/// format change, draining the outputs.
fn decode_vp9_with_buffers(
    decoder: &mut StatelessDecoder<Vp9, HostBackend>,
    timestamp: u64,
    unit: &[u8],
    buffers: usize,
) -> Result<usize, DecodeError> {
    let mut left = buffers;
    loop {
        let result = decoder.decode(timestamp, unit, &mut || {
            (left > 0).then(|| {
                left -= 1;
                HostFrame
            })
        });
        while decoder.next_event().is_some() {}
        if !matches!(result, Err(DecodeError::CheckEvents)) {
            return result;
        }
    }
}

/// A superframe short of output buffers asks for the pictures it still needs: its
/// coded frames less the ones it allocated, wherever its show-existing frames sit.
/// Given that many more buffers, it decodes.
#[test]
fn vp9_a_superframe_short_of_output_buffers_asks_for_its_coded_frames_only() {
    let units = ivf_units(include_bytes!(
        "../../codec/vp9/test_data/vp90-2-10-show-existing-frame.vp9.ivf"
    ));
    let key_frame = units[0].as_slice();
    let show_slot_0 = VP9_SHOW_SLOT_0.as_slice();
    let mut decoder =
        StatelessDecoder::<Vp9, HostBackend>::new(HostBackend::default(), BlockingMode::Blocking)
            .expect("a decoder on the host backend");
    // The key frame fills every reference slot.
    let decoded = decode_vp9_with_buffers(&mut decoder, 0, key_frame, 1).expect("the key frame");
    assert_eq!(decoded, key_frame.len());

    // (frames, buffers given, pictures still needed)
    let superframes: [(&[&[u8]], usize, usize); 3] = [
        (&[show_slot_0, show_slot_0, key_frame], 0, 1),
        (&[show_slot_0, key_frame], 0, 1),
        (&[key_frame, show_slot_0, key_frame], 1, 1),
    ];
    for (timestamp, (frames, buffers, needed)) in (1..).zip(superframes) {
        let superframe = vp9_superframe(frames);
        match decode_vp9_with_buffers(&mut decoder, timestamp, &superframe, buffers) {
            Err(DecodeError::NotEnoughOutputBuffers(n)) => {
                assert_eq!(n, needed, "superframe {timestamp} with {buffers} buffers")
            }
            other => panic!("superframe {timestamp} with {buffers} buffers: {other:?}"),
        }
        let given = buffers + needed;
        let decoded = decode_vp9_with_buffers(&mut decoder, timestamp, &superframe, given)
            .unwrap_or_else(|e| panic!("superframe {timestamp} with {given} buffers: {e}"));
        assert_eq!(decoded, superframe.len());
    }
}

/// An encoder's hidden alternate references: each is decoded in one temporal unit
/// and shown by a show-existing frame header in a later one, which is its output's
/// timestamp.
#[test]
fn av1_hidden_pictures_come_out_under_the_unit_that_shows_them() {
    let units =
        ivf_units(include_bytes!("../../codec/av1/test_data/show-existing-frame-10bit.ivf"));
    assert_eq!(units.len(), 30);
    let outputs = decode_units::<Av1>(&units);
    assert_one_output_per_unit(&outputs, units.len());
    assert!(shown_again(&outputs).is_empty(), "each picture is shown once");
    let shown_later: Vec<usize> = outputs
        .iter()
        .enumerate()
        .filter(|(_, h)| h.picture.decoded_at < h.timestamp)
        .map(|(unit, _)| unit)
        .collect();
    assert_eq!(shown_later.len(), 14, "the stream's 14 show-existing frames: {shown_later:?}");
    for (unit, handle) in outputs.iter().enumerate() {
        assert!(
            handle.picture.decoded_at <= unit as u64,
            "unit {unit} shows a picture decoded later"
        );
    }
}

/// A shown inter picture shown again by an inserted temporal unit: the second
/// output shares the first's picture and carries its own unit's timestamp.
#[test]
fn av1_a_shown_picture_shown_again_comes_out_under_the_showing_unit() {
    let mut units = ivf_units(include_bytes!("../../codec/av1/test_data/test-25fps.av1.ivf"));
    units.truncate(12);
    // A temporal delimiter, then a frame header OBU: show_existing_frame = 1,
    // frame_to_show_map_idx = 1 (unit 2's picture, `refresh_frame_flags` 0b10),
    // trailing bits.
    units.insert(3, vec![0x12, 0x00, 0x1a, 0x01, 0x98]);
    let outputs = decode_units::<Av1>(&units);
    assert_one_output_per_unit(&outputs, units.len());
    assert_eq!(shown_again(&outputs), [(2, 3)]);
    assert_eq!(outputs[2].timestamp, 2);
    assert_eq!(outputs[3].picture.decoded_at, 2);
}
