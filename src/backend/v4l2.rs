// Copyright 2024 The ChromiumOS Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

//! V4L2 backend

use v4l2r::nix::sys::time::TimeVal;

pub mod decoder;
pub mod encoder;

impl From<v4l2r::PixelFormat> for crate::Fourcc {
    fn from(value: v4l2r::PixelFormat) -> Self {
        crate::Fourcc(value.to_u32())
    }
}

impl From<crate::Fourcc> for v4l2r::PixelFormat {
    fn from(value: crate::Fourcc) -> Self {
        v4l2r::PixelFormat::from_u32(value.0)
    }
}

/// A buffer timestamp as V4L2 carries it: microseconds, in a `timeval`.
///
/// The kernel keeps a buffer's timestamp as `tv_sec * 10^9 + tv_usec * 10^3`
/// nanoseconds, reading `tv_usec` as a `u32`, and splits it back into seconds and
/// microseconds when the buffer is dequeued. Split the same way when queued, a
/// timestamp comes back unchanged, and the kernel knows the buffer by `timestamp * 1000`
/// nanoseconds: the value `v4l2_timeval_to_ns` gives, which the stateless decoders'
/// reference controls name.
#[repr(transparent)]
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Timestamp(pub u64);

impl From<v4l2r::bindings::timeval> for Timestamp {
    fn from(value: v4l2r::bindings::timeval) -> Self {
        let timestamp = value.tv_sec.wrapping_mul(1_000_000);
        let timestamp = timestamp.wrapping_add(value.tv_usec);
        Timestamp(timestamp.max(0) as u64)
    }
}

impl From<&Timestamp> for TimeVal {
    fn from(value: &Timestamp) -> Self {
        let tv_sec = (value.0 / 1_000_000).min(i64::MAX as u64);
        let tv_usec = (value.0 % 1_000_000).min(i64::MAX as u64);
        Self::new(tv_sec as i64, tv_usec as i64)
    }
}

#[cfg(test)]
mod timestamp_tests;
