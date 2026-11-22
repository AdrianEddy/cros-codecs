// Copyright 2026 The ChromiumOS Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

//! The H.264 decoder starts a new sequence (`new_sequence`, after draining) for
//! an SPS whose `NegotiationInfo` differs from the running one's. A backend sizes
//! its reference storage by the sequence it is given, so an SPS that holds more
//! reference frames at the same format must differ, though without the VUI's
//! `max_dec_frame_buffering` its `max_dpb_frames` is the level's bound, the same
//! for any reference count the level holds. So must an SPS that codes its
//! colour planes apart at the same `chroma_format_idc`, which changes what a
//! picture is.

use std::rc::Rc;

use super::NegotiationInfo;
use crate::codec::h264::parser::Level;
use crate::codec::h264::parser::Profile;
use crate::codec::h264::parser::Sps;
use crate::codec::h264::parser::SpsBuilder;

/// A High 4:2:0 8-bit SPS of 320 × 240 frames at level 3.0 holding
/// `max_num_ref_frames` references, its VUI stating `max_dec_frame_buffering`
/// when given one.
fn sps(max_num_ref_frames: u8, max_dec_frame_buffering: Option<u32>) -> Sps {
    let mut builder = SpsBuilder::new()
        .profile_idc(Profile::High)
        .level_idc(Level::L3)
        .resolution(320, 240)
        .chroma_format_idc(1)
        .frame_mbs_only_flag(true)
        .max_num_ref_frames(max_num_ref_frames);
    if max_dec_frame_buffering.is_some() {
        builder = builder.vui_parameters_present();
    }
    let mut sps = Rc::into_inner(builder.build()).expect("a fresh SPS");
    if let Some(frames) = max_dec_frame_buffering {
        sps.vui_parameters.bitstream_restriction_flag = true;
        sps.vui_parameters.max_dec_frame_buffering = frames;
    }
    sps
}

#[test]
fn an_sps_that_holds_more_reference_frames_starts_a_sequence() {
    // No `max_dec_frame_buffering`: the level's bound, whatever the count.
    let (one, four) = (sps(1, None), sps(4, None));
    assert_eq!(one.max_dpb_frames(), four.max_dpb_frames(), "the level bounds both");
    assert_ne!(NegotiationInfo::from(&one), NegotiationInfo::from(&four));

    // The VUI's buffering moves with the count.
    let (one, four) = (sps(1, Some(1)), sps(4, Some(4)));
    assert_ne!(one.max_dpb_frames(), four.max_dpb_frames());
    assert_ne!(NegotiationInfo::from(&one), NegotiationInfo::from(&four));

    // The same SPS again keeps the sequence.
    assert_eq!(NegotiationInfo::from(&sps(4, None)), NegotiationInfo::from(&sps(4, None)));
    assert_eq!(NegotiationInfo::from(&sps(4, Some(4))), NegotiationInfo::from(&sps(4, Some(4))));
}

#[test]
fn an_sps_that_codes_its_colour_planes_apart_starts_a_sequence() {
    let mut together = sps(1, None);
    together.chroma_format_idc = 3;
    let mut apart = sps(1, None);
    apart.chroma_format_idc = 3;
    apart.separate_colour_plane_flag = true;
    assert_ne!(NegotiationInfo::from(&together), NegotiationInfo::from(&apart));
}
