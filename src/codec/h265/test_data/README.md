# H.265 Test Data

This document lists the test data used by the H.265 parser.

## bear.hevc

Same as Chromium's `bbb.hevc`.

## bear.hevc

Same as Chromium's `bear.hevc`.

## test-25fps.hevc

Same as Chromium's `test-25fps.hevc`.

The slice data for the first two slices in this stream was extracted manually from GStreamer using GDB.

## wpp-64x1088-ctb16.h265

One 64x1088 IDR picture with wavefront parallel processing and 16x16 CTBs, so
its slice header codes 67 entry points (one per CTB row after the first).
Generated with FFmpeg N-125350 and libx265:

```
ffmpeg -f lavfi -i "testsrc=size=64x1088:rate=1:duration=1" -frames:v 1 \
    -c:v libx265 -preset ultrafast \
    -x265-params "ctu=16:min-cu-size=8:wpp=1:frame-threads=1:annexb=1:repeat-headers=1" \
    -f hevc wpp-64x1088-ctb16.h265
```
