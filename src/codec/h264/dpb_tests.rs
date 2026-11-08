// Copyright 2026 The ChromiumOS Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

//! Memory management control operations that name a picture the DPB does not
//! hold: each reports the lookup that failed and leaves the DPB as it was, which
//! is what lets the decoder skip it.

use super::Dpb;
use super::MmcoError;
use crate::codec::h264::parser::RefPicMarkingInner;
use crate::codec::h264::picture::PictureData;
use crate::codec::h264::picture::RcPictureData;
use crate::codec::h264::picture::Reference;

/// A DPB holding one frame marked `reference`, with both its picture numbers at
/// `num`.
fn dpb_with(reference: Reference, num: u32) -> (Dpb<()>, RcPictureData) {
    let mut pic = PictureData::default();
    pic.pic_num = num as i32;
    pic.long_term_pic_num = num;
    pic.set_reference(reference, false);
    let pic = pic.into_rc();
    let mut dpb = Dpb::default();
    dpb.set_limits(4, 0);
    dpb.store_picture(pic.clone(), None).expect("room for one picture");
    (dpb, pic)
}

#[test]
fn mmco_2_without_that_long_term_picture_reports_a_long_term_lookup() {
    // A short-term picture with the same number is not the long-term one MMCO 2
    // names.
    let (mut dpb, held) = dpb_with(Reference::ShortTerm, 3);
    let marking = RefPicMarkingInner {
        memory_management_control_operation: 2,
        long_term_pic_num: 3,
        ..Default::default()
    };

    let err = dpb.mmco_op_2(&PictureData::default(), &marking).expect_err("no long-term picture 3");

    assert!(matches!(err, MmcoError::NoLongTermPic), "{err:?}");
    assert!(err.to_string().contains("LongTerm"), "{err}");
    assert_eq!(*held.borrow().reference(), Reference::ShortTerm, "the DPB is as it was");
}

#[test]
fn mmco_1_without_that_short_term_picture_reports_a_short_term_lookup() {
    // The current picture's pic_num 3 minus (0 + 1) names pic_num 2, which only a
    // long-term picture carries.
    let (mut dpb, held) = dpb_with(Reference::LongTerm, 2);
    let mut current = PictureData::default();
    current.pic_num = 3;
    let marking = RefPicMarkingInner {
        memory_management_control_operation: 1,
        difference_of_pic_nums_minus1: 0,
        ..Default::default()
    };

    let err = dpb.mmco_op_1(&current, &marking).expect_err("no short-term picture 2");

    assert!(matches!(err, MmcoError::NoShortTermPic), "{err:?}");
    assert_eq!(*held.borrow().reference(), Reference::LongTerm, "the DPB is as it was");
}
