Play the source video with the current frame's gaze dot (yellow), left PSM mask
centroid (cyan), and right PSM mask centroid (orange):

```bash
conda activate sam3
python tools/play_gaze_psm.py
```

Defaults select novice Task A and C recordings from `Index/content_index.csv`.
Use the recording selector or Previous/Next video buttons to change recordings.
The stage selector jumps directly to any consecutive occurrence; repeated stages
and `none` intervals remain separate entries. Playback automatically updates the
stage when it crosses a boundary.

| Control | Action |
| --- | --- |
| `<` or `,` | Previous frame; pauses playback |
| `>` or `.` | Next frame; pauses playback |
| Left arrow / Right arrow | Previous / next stage; pauses at its first frame |
| Space | Play / pause |
| Timeline slider | Seek to a frame |
| Stage selector | Jump to the selected stage occurrence |
| Speed selector | 0.25×, 0.5×, 1×, or 2× playback |
| Show mask regions | Tint the actual current-frame mask pixels |
| Esc | Close |

Frame indices in the timeline are zero based. The status line also shows the
original CSV `Frame_Number`, capture time relative to the first retained frame,
stage, and overlay coordinates in original-resolution pixels. Arrow shortcuts
also work when the timeline or a selector has keyboard focus. At the first/last
stage, navigation past the recording boundary does nothing.

To open a particular recording, including a different skill or task:

```bash
python tools/play_gaze_psm.py --records User1_1_1
python tools/play_gaze_psm.py --tasks C --skill novice
python tools/play_gaze_psm.py --tasks A C --skill all --list
python tools/play_gaze_psm.py --data-root /path/to/dvss
```

Gaze defaults to raw `csv/<id>_gaze.csv` samples with `Status=2`, normalized
`norm_x`/`norm_y`, and exact `Capture Unix timestamp` alignment to
`csv/<id>_video_ts.csv`. It interpolates valid raw samples only across gaps of at
most 100 ms (`--max-gap`); it applies no smoothing or lag shift. Invalid or missing
gaze is shown as unavailable. To display the indexed per-frame gaze instead:

```bash
python tools/play_gaze_psm.py --gaze-source allocation
```

Allocation gaze uses `npy/<id>_gaze.csv`, matching rows by `Frame_Number` and
accepting `valid=2`; invalid allocation frames stay blank. `--gaze-units pixels`,
`--gaze-x-column`, `--gaze-y-column`, `--gaze-valid-values`, and `--flip-gaze-y`
support alternate gaze conventions.

Both centroids are calculated directly from the current frames of
`sam/<id>/mask/obj_psm_l.mp4` and `obj_psm_r.mp4`, using grayscale >127 and
all foreground pixels. Masks with fewer than 10 pixels have no centroid.
`--threshold` and `--min-area` change these settings. Left/right follows the
source filenames. The player validates mask counts and dimensions before
displaying overlays. It handles RGB videos with or without deleted timestamp
rows while masks must follow the retained timeline.

Requires a desktop display, Tkinter, Pillow, OpenCV, NumPy, and the existing
analysis dependencies (SciPy). It streams frames and never saves changes to
videos, masks, gaze CSVs, or analysis outputs.
