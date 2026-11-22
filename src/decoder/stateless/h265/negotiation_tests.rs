// Copyright 2026 The ChromiumOS Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

//! The H.265 decoder starts a new sequence (`new_sequence`, after draining) for
//! an SPS whose `NegotiationInfo` differs from the running one's, or whose
//! level gives another `MaxDpbSize`. A backend sizes its reference storage by
//! the sequence it is given, so an SPS that raises
//! `sps_max_dec_pic_buffering_minus1` at the same level and format must differ:
//! the level's `MaxDpbSize`, this decoder's own DPB size, stays. So must an SPS
//! that codes its colour planes apart at the same `chroma_format_idc`.
//!
//! The SPS is `test-25fps.h265`'s (Main, 320 × 240, one sub-layer).

use std::io::Cursor;

use super::NegotiationInfo;
use crate::codec::h264::nalu::Header;
use crate::codec::h265::parser::Nalu;
use crate::codec::h265::parser::NaluType;
use crate::codec::h265::parser::Parser;
use crate::codec::h265::parser::Sps;

const STREAM: &[u8] = include_bytes!("../../../codec/h265/test_data/test-25fps.h265");

/// The stream's first SPS, its VPS parsed ahead of it.
fn stream_sps() -> Sps {
    let mut parser = Parser::default();
    let mut cursor = Cursor::new(STREAM);
    while let Ok(nalu) = Nalu::next(&mut cursor) {
        match nalu.header.type_ {
            NaluType::VpsNut => {
                parser.parse_vps(&nalu).expect("the stream's VPS");
            }
            NaluType::SpsNut => return parser.parse_sps(&nalu).expect("the stream's SPS").clone(),
            _ => {}
        }
    }
    panic!("the stream carries no SPS");
}

#[test]
fn an_sps_that_needs_a_larger_decoded_picture_buffer_starts_a_sequence() {
    let sps = stream_sps();
    let top = usize::from(sps.max_sub_layers_minus1);
    let mut larger = sps.clone();
    larger.max_dec_pic_buffering_minus1[top] += 3;
    assert_eq!(sps.max_dpb_size(), larger.max_dpb_size(), "the level's MaxDpbSize is the same");
    assert_ne!(NegotiationInfo::from(&sps), NegotiationInfo::from(&larger));
    assert_eq!(NegotiationInfo::from(&sps), NegotiationInfo::from(&sps.clone()));
}

#[test]
fn an_sps_that_codes_its_colour_planes_apart_starts_a_sequence() {
    let mut together = stream_sps();
    together.chroma_format_idc = 3;
    let mut apart = together.clone();
    apart.separate_colour_plane_flag = true;
    assert_ne!(NegotiationInfo::from(&together), NegotiationInfo::from(&apart));
}

/// `sps_max_sub_layers_minus1` and `vps_max_sub_layers_minus1` are 0..=6, and
/// every per-sub-layer table is indexed by them, `NegotiationInfo`'s
/// `max_dec_pic_buffering_minus1` among them: the parser refuses a 7 instead of
/// indexing past the tables.
#[test]
fn a_parameter_set_naming_eight_sub_layers_is_refused() {
    let mut parser = Parser::default();
    let mut cursor = Cursor::new(STREAM);
    let (mut vps_refused, mut sps_refused) = (false, false);
    while let Ok(nalu) = Nalu::next(&mut cursor) {
        // The unit behind a start code again, its payload after the header.
        let payload = 4 + nalu.header.len();
        let mut bytes = vec![0, 0, 0, 1];
        bytes.extend_from_slice(nalu.as_ref());
        match nalu.header.type_ {
            // vps_video_parameter_set_id (4 bits), base_layer_internal_flag,
            // base_layer_available_flag, vps_max_layers_minus1 (6), then
            // vps_max_sub_layers_minus1 (3): the second byte's 0b0000_1110.
            NaluType::VpsNut => bytes[payload + 1] |= 0b0000_1110,
            // sps_video_parameter_set_id (4 bits), then
            // sps_max_sub_layers_minus1 (3): the first byte's 0b0000_1110.
            NaluType::SpsNut => bytes[payload] |= 0b0000_1110,
            _ => continue,
        }
        let mut patched = Cursor::new(&bytes[..]);
        let patched = Nalu::next(&mut patched).expect("the patched unit frames");
        match patched.header.type_ {
            NaluType::VpsNut => {
                let e = parser.parse_vps(&patched).expect_err("eight sub-layers");
                assert!(e.contains("vps_max_sub_layers_minus1"), "{e}");
                vps_refused = true;
            }
            _ => {
                let e = parser.parse_sps(&patched).expect_err("eight sub-layers");
                assert!(e.contains("sps_max_sub_layers_minus1"), "{e}");
                sps_refused = true;
                break;
            }
        }
    }
    assert!(vps_refused && sps_refused, "the stream's VPS and SPS were both patched");
}
