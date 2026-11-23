// Copyright 2024 The ChromiumOS Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

//! Hardware test of the packed slice header, ignored by default as the crate's other VA-API
//! tests are: on a driver that advertises `VA_ENC_PACKED_HEADER_SLICE`, the stream the driver
//! writes carries, for every I and P slice, the NAL unit header and the picture counters the
//! backend packed. A driver that wrote its own slice header instead (radeonsi without a packed
//! one writes `nal_ref_idc` 0, `nal_unit_type` 0, `frame_num` 0 and POC 0) fails it.

use std::collections::BTreeMap;
use std::io::Cursor;
use std::rc::Rc;

use libva::Display;
use libva::UsageHint;
use libva::VAEntrypoint::VAEntrypointEncSlice;
use libva::VAEntrypoint::VAEntrypointEncSliceLP;
use libva::VAProfile::VAProfileH264Main;
use libva::VA_RT_FORMAT_YUV420;

use crate::backend::vaapi::encoder::query_packed_headers;
use crate::backend::vaapi::encoder::tests::TestFrameGenerator;
use crate::backend::vaapi::encoder::VaapiBackend;
use crate::backend::vaapi::surface_pool::PooledVaSurface;
use crate::backend::vaapi::surface_pool::VaSurfacePool;
use crate::codec::h264::parser::Nalu;
use crate::codec::h264::parser::NaluType;
use crate::codec::h264::parser::Parser;
use crate::codec::h264::parser::Profile;
use crate::codec::h264::parser::SliceHeader;
use crate::decoder::FramePool;
use crate::encoder::h264::EncoderConfig;
use crate::encoder::simple_encode_loop;
use crate::encoder::stateless::h264::StatelessEncoder;
use crate::encoder::PredictionStructure;
use crate::encoder::RateControl;
use crate::encoder::Tunings;
use crate::BlockingMode;
use crate::FrameLayout;
use crate::PlaneLayout;
use crate::Resolution;

const WIDTH: usize = 320;
const HEIGHT: usize = 240;
/// Long enough that `frame_num` and the POC lsb wrap inside the first GOP (MaxFrameNum 32).
const GOP: usize = 40;
/// Two GOPs: the second IDR must step `idr_pic_id`.
const FRAMES: usize = 45;

#[test]
#[ignore]
fn a_driver_that_takes_packed_slice_headers_codes_the_ones_packed() {
    let display = Display::open().expect("no VA display");
    let entrypoints = display.query_config_entrypoints(VAProfileH264Main).unwrap();
    let low_power = entrypoints.contains(&VAEntrypointEncSliceLP);
    let entrypoint = if low_power { VAEntrypointEncSliceLP } else { VAEntrypointEncSlice };
    let packed = query_packed_headers(&display, VAProfileH264Main, entrypoint);
    assert_ne!(
        packed & libva::VA_ENC_PACKED_HEADER_SLICE,
        0,
        "the driver takes no packed slice header (mask {packed:#x}); this test needs one that does"
    );

    let config = EncoderConfig {
        profile: Profile::Main,
        resolution: Resolution { width: WIDTH as u32, height: HEIGHT as u32 },
        pred_structure: PredictionStructure::LowDelay { limit: GOP as u16 },
        initial_tunings: Tunings {
            rate_control: RateControl::ConstantBitrate(1_200_000),
            framerate: 30,
            ..Default::default()
        },
        // As `StatelessEncoder::new_vaapi` decides it: the driver writes the packed parameter sets.
        emit_parameter_sets: false,
        ..Default::default()
    };
    let frame_layout = FrameLayout {
        format: (b"NV12".into(), 0),
        size: config.resolution,
        planes: vec![
            PlaneLayout { buffer_index: 0, offset: 0, stride: WIDTH },
            PlaneLayout { buffer_index: 0, offset: WIDTH * HEIGHT, stride: WIDTH },
        ],
    };
    let backend = VaapiBackend::<(), PooledVaSurface<()>>::new_with_packed_headers(
        Rc::clone(&display),
        VAProfileH264Main,
        frame_layout.format.0,
        frame_layout.size,
        libva::VA_RC_CBR,
        low_power,
        packed,
    )
    .unwrap();
    let mut encoder = StatelessEncoder::<PooledVaSurface<()>, _>::new_h264(
        backend,
        config,
        BlockingMode::Blocking,
    )
    .unwrap();

    let mut pool = VaSurfacePool::new(
        Rc::clone(&display),
        VA_RT_FORMAT_YUV420,
        Some(UsageHint::USAGE_HINT_ENCODER),
        frame_layout.size,
    );
    pool.add_frames(vec![(); 16]).unwrap();
    let mut frames =
        TestFrameGenerator::new(FRAMES as u64, display, pool, frame_layout).take(FRAMES);
    let mut bitstream = Vec::new();
    simple_encode_loop(&mut encoder, &mut frames, |coded| bitstream.extend(coded.bitstream))
        .unwrap();

    // Read back what the driver wrote, with the parameter sets it wrote.
    let mut parser = Parser::default();
    let mut cursor = Cursor::new(&bitstream[..]);
    // Each NAL unit header byte the stream holds, and how many times.
    let mut headers = BTreeMap::new();
    let mut slices: Vec<((u8, NaluType), SliceHeader)> = Vec::new();
    while let Ok(nalu) = Nalu::next(&mut cursor) {
        *headers.entry((nalu.header.ref_idc << 5) | nalu.header.type_ as u8).or_insert(0) += 1;
        match nalu.header.type_ {
            NaluType::Sps => {
                parser.parse_sps(&nalu).unwrap();
            }
            NaluType::Pps => {
                parser.parse_pps(&nalu).unwrap();
            }
            NaluType::Slice | NaluType::SliceIdr => {
                let nal = (nalu.header.ref_idc, nalu.header.type_);
                slices.push((nal, parser.parse_slice_header(nalu).unwrap().header));
            }
            _ => {}
        }
    }
    let counts: Vec<String> = headers.iter().map(|(byte, n)| format!("{byte:02x} x{n}")).collect();
    eprintln!("[packed-slice] {} slices; NAL unit headers {}", slices.len(), counts.join(", "));
    assert_eq!(slices.len(), FRAMES, "one coded slice per frame");
    let sps = parser.get_sps(0).expect("the stream states no SPS");
    let (max_frame_num, max_lsb) = (sps.max_frame_num(), sps.max_pic_order_cnt_lsb());
    assert!(max_frame_num < GOP as u32, "MaxFrameNum {max_frame_num}: the GOP must cross it");

    for (n, (nal, header)) in slices.iter().enumerate() {
        let in_gop = n % GOP;
        let idr = in_gop == 0;
        let want_nal = if idr { (3, NaluType::SliceIdr) } else { (2, NaluType::Slice) };
        assert_eq!(*nal, want_nal, "frame {n}'s NAL unit header");
        assert_eq!(header.slice_type.is_i(), idr, "frame {n} is an I slice exactly when an IDR");
        assert_eq!(header.slice_type.is_p(), !idr, "frame {n} is a P slice otherwise");
        assert_eq!(
            u32::from(header.frame_num),
            in_gop as u32 % max_frame_num,
            "frame {n}'s frame_num"
        );
        assert_eq!(
            u32::from(header.pic_order_cnt_lsb),
            2 * in_gop as u32 % max_lsb,
            "frame {n}'s POC lsb"
        );
        if idr {
            assert_eq!(usize::from(header.idr_pic_id), n / GOP, "frame {n}'s idr_pic_id");
        }
    }
}
