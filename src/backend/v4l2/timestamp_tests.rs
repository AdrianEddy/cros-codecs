// Copyright 2026 The ChromiumOS Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

//! A timestamp queued on a V4L2 buffer comes back unchanged when the buffer is
//! dequeued, and the kernel knows the buffer by `timestamp * 1000` nanoseconds.
//!
//! The kernel side follows `include/media/v4l2-common.h`: queuing keeps
//! `tv_sec * 10^9 + (u32)tv_usec * 10^3` nanoseconds (`v4l2_buffer_get_timestamp`),
//! and dequeuing splits them back into seconds and microseconds
//! (`v4l2_buffer_set_timestamp`, through `ns_to_timespec64`).

use v4l2r::bindings::timeval;
use v4l2r::nix::sys::time::TimeVal;

use super::Timestamp;

/// Timestamps on each side of what the round trip must survive: a second, the
/// `u32` the kernel reads `tv_usec` as, a day of microseconds, and the largest
/// timestamp whose nanoseconds the kernel splits back (`ns_to_timespec64` takes
/// an `s64`).
const TIMESTAMPS: [u64; 8] = [
    0,
    999_999,
    1_000_000,
    1_234_567,
    u32::MAX as u64,
    u32::MAX as u64 + 1,
    86_400_000_000,
    i64::MAX as u64 / 1000,
];

/// The nanoseconds the kernel keeps for a buffer queued with `queued`.
fn kernel_timestamp_ns(queued: &TimeVal) -> u64 {
    queued.tv_sec() as u64 * 1_000_000_000 + u64::from(queued.tv_usec() as u32) * 1_000
}

/// The `timeval` the kernel fills in when it dequeues a buffer it keeps `ns` for.
fn dequeued(ns: u64) -> timeval {
    timeval { tv_sec: (ns / 1_000_000_000) as _, tv_usec: (ns % 1_000_000_000 / 1_000) as _ }
}

#[test]
fn a_timestamp_is_queued_as_seconds_and_microseconds() {
    let queued = TimeVal::from(&Timestamp(1_234_567));
    assert_eq!((queued.tv_sec(), queued.tv_usec()), (1, 234_567));
    for timestamp in TIMESTAMPS {
        let queued = TimeVal::from(&Timestamp(timestamp));
        assert!(queued.tv_usec() < 1_000_000, "{timestamp}: tv_usec {}", queued.tv_usec());
    }
}

#[test]
fn a_timestamp_comes_back_from_a_dequeued_buffer_unchanged() {
    for timestamp in TIMESTAMPS {
        let ns = kernel_timestamp_ns(&TimeVal::from(&Timestamp(timestamp)));
        assert_eq!(ns, timestamp * 1000, "{timestamp}: the nanoseconds a reference names");
        assert_eq!(Timestamp::from(dequeued(ns)), Timestamp(timestamp));
    }
}
