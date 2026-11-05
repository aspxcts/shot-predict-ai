# Tennis shot lab: usage guide

> Detailed guide and experiment log. For the project overview, see the [README](../README.md).
> Paths under `data\` and `runs\` refer to local artifacts (footage, labels, checkpoints,
> experiment scripts) that are not published in the repository.

A local first version of a tennis placement predictor: a video labeling app,
ResNet18 + GRU training, and rolling prediction-video export. No trained tennis
weights or footage are included. Predictions will not be meaningful until you
collect representative data and demonstrate held-out performance.

## Start on Windows

Python 3.12+ is required. A project-local Python 3.12 environment is already
created on the development machine. For a fresh installation:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

Launch the labeling app:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1 --browser.gatherUsageStats false
```

Open `http://127.0.0.1:8501`. Stop the server with Ctrl+C. The app is intended
for one local annotator; do not expose it publicly or edit the same project
in simultaneous sessions. Imported paths, optional player names, footage, and
labels stay on your machine. No external inference service is used.
After updating the app's Python files, stop and restart an existing server;
refreshing the browser alone may retain old imported modules.

## The easy data-collection workflow

1. **Import a match video** you have permission to use from the sidebar. For
   a large video, paste its local path instead of uploading/copying it. Use
   the same match ID for every clip from one match. Optional player names
   help identify train/validation player overlap.
2. **Calibrate the view:** click near-left, near-right, far-right, then
   far-left **singles** baseline corners as seen on screen. All four clicks
   must use the same frame. Check that the green grid follows the court.
3. Use the video, timeline, frame buttons, or exact-frame field to find
   racket-ball contact. Choose groundstroke/serve, then click **Near** or
   **Far** to mark the hitter and contact in one action.
4. Advance to the **first bounce**. Click anywhere inside its correct green
   receiving-court zone; the click automatically saves the label. An exact
   ball click is unnecessary for six-zone training, though it provides a
   more precise stored landing coordinate for future work.
5. Continue labeling. **Undo last** removes the most recent label for the
   current clip; the sidebar also offers label review, deletion, and counts.

The player and scrubber share one frame clock. Space plays/pauses, left/right
arrows or the mouse wheel step one frame, Shift plus arrows/wheel moves one
second, and J/L moves five seconds. N/F marks a near/far hit, G/S selects
groundstroke/serve, and Ctrl+Z undoes the last saved label. Use a current
Chrome or Edge and a constant-frame-rate H.264 MP4 for reliable browser
decoding. If the browser lacks displayed-frame callbacks, the app warns that
frame accuracy may be reduced.

You do **not** need bounding boxes for players, rackets, or every ball frame
to train this baseline. Each example requires just contact timing, bounce
timing/position, and two categorical choices.

### Collection rules

- Start with normal-speed baseline rallies; add serves after the workflow
  feels comfortable. Include both near and far hitters.
- Skip replays, slow motion, camera cuts, volleys, overheads, out balls,
  net shots, and bounces you cannot confidently locate. These classes are
  intentionally unsupported by this version.
- If framing changes, use **Recalibrate** and check the new grid before
  saving more shots. Existing labels retain their recorded calibration.
  The current YOLO crop tool uses the clip's current calibration rather than
  each shot's historical snapshot; split substantially different views into
  fixed-view clips and inspect its crop audits. Keep the same match ID and
  avoid duplicate footage under different match IDs.
- Start with a few dozen shots to exercise the pipeline, then collect
  hundreds/thousands across varied matches. This is not an accuracy
  guarantee or a prescribed sufficient dataset size.
- Collect naturally occurring placements, not just obvious winners. Monitor
  missing zones, hitter imbalance, and shot-type imbalance.
- Reserve complete matches for validation. To assess unfamiliar-player
  performance, reserve matches whose players never appear in training.
- Videos need constant frame rate for reliable frame-time conversion.
  Prefer original normal-speed H.264 MP4 files; convert variable-frame-rate
  footage before importing.

### Zone definitions

Each receiving half-court is split into three equal width thirds and two
equal depth halves. **Deep** is closer to the receiving baseline; **short**
is closer to the net. Ad/deuce are from the **receiver's perspective facing
the net**, not fixed screen left/right:

| Receiving player | Screen left | Middle | Screen right |
| --- | --- | --- | --- |
| Far | Deuce | Middle | Ad |
| Near | Ad | Middle | Deuce |

The class order is `ad_deep`, `middle_deep`, `deuce_deep`, `ad_short`,
`middle_short`, `deuce_short`. These regions are not the service boxes.
Serve labels describe actual first-bounce regions using the same grid.

Labels and calibration are saved atomically in `data\project.json`.
Existing videos are referenced by absolute path: moving them requires
updating those references. Back up the project file and source footage.
The app also keeps a content-addressed copy in `data\backups\` (one per
distinct project file, newest 20) each time the server starts.

### Semi-automated labeling (review queue)

An offline analysis proposes shots; you confirm or correct them instead of
hunting through the video. The analysis **never reads or writes labels**.

```powershell
# 1. Register the court once per camera (US Open's main camera is already registered in
#    data\camera\references.json; its corners are an unconfirmed estimate: check the green grid).
# 2. Analyze clips one at a time (resumable; about 10 minutes per 5-minute 60 fps clip on this
#    CPU-only machine). For many clips use the parallel `batch` command in "Validate mode" below.
.\.venv\Scripts\python.exe -m shot_predict.autopilot.cli analyze --match-id paul_alcaraz_us_open_2026 --clips 3
.\.venv\Scripts\python.exe -m shot_predict.autopilot.cli status
# 3. Restart Streamlit (Ctrl+C, then your usual command) and open an analyzed clip.
```

The review queue is on by default when a clip has proposals. Per proposal:
**Contact?** jumps to the proposed contact frame with player boxes and the ball
trail (the yellow line is only this shot's own flight: the incoming ball and the
outgoing ball followed through its bounce; Enter accepts, arrows nudge, N/F hitter,
G/S type) and **Bounce?**
jumps to the proposed bounce (click where the ball landed; Shift+`,` / `.`
nudge five frames). `X` not a shot, `K` out/net, `Q` skip, `P` slow-motion
preview, `[` `]` browse, `Ctrl+Z` undo, `M` manual labeling (the original
workflow, unchanged).

- **Weak proposals are never one-key.** Below 50% confidence Enter is
  disabled until you set the hitter yourself. A proposed bounce is **always** a
  hint that requires a click: measured on labeled Turin shots, proposed bounces
  were early and rarely gave the right zone, so their confidence is capped at 45%
  (`events.BOUNCE_CONFIDENCE_CAP`). Raise it only after measuring a better engine.
- **Provenance** is stored on every queue-made label: `source` (`auto_confirmed`
  or `auto_corrected`), the original `proposal` (without the bulky ball trail),
  `edited_fields`, `review_seconds`, and the court corners at the input frames
  (`input_court_corners`) for moving cameras. Rejected / out / skipped proposals
  are remembered in `reviews` and not queued again. Filter by `source` when you
  want to train on human-only labels.
- **Moving cameras** (US Open) use per-frame court corners recovered by matching
  the camera's background to a reference frame whose corners you registered
  (`data\camera\references.json`; add more with `... cli add-reference`). Only
  views related to a registered camera are analyzed: close-ups, replays, crowd
  shots, and *other* cameras (for example the low behind-the-baseline camera) are
  skipped, and are not labelable until you register that camera.
- Score the engine against your own labels with
  `python -m shot_predict.autopilot.evaluate --analysis <folder> --match-id <id> --output <new folder>`.

Measured on labeled Turin rallies (`runs\autopilot-eval\report-lead`, 58 shots; two of
the three blocks were never used to tune the engine): contact within 200 ms of
your label for 81% of shots (64% within 100 ms, 88% within 500 ms; median error 40 ms),
hitter correct for 47/47 paired shots, about 0.26 spurious proposals per paired shot.
Proposed bounces are within tolerance for only 8 of 47 and landing zones agree with yours
for 22 of 47, because a proposal without a landing zone counts as wrong there. These are
same-match numbers from a color-agnostic motion tracker; they do **not** establish US Open
accuracy. The strict gates (contact within 100 ms for 80%, zone agreement 70%) are still
not met. Far-hitter contacts are moved 0.12 s earlier because they measured late by a
median of 160 ms on Turin and 67-133 ms on your US Open edits; near-hitter contacts were
unbiased. Court lines and the net are deliberately not masked out of the ball search:
both a pixel mask and a track-level filter cut Turin recall (79% to 57% and to 72%
within 200 ms) for fewer false tracklets.

### Validate mode: label tens of matches by confirming, not clicking

Validate mode is for volume. The machine proposes every shot with its landing zone; you
watch a short loop and answer **Correct** or **Wrong**. No clicking on the court, no
scrubbing. Cards continue from clip to clip by themselves.

```powershell
# 1. Register a match once: a full recording is split into ~5-minute clips (stream copy,
#    fast and lossless); a folder of clips you already have is used in place.
.\.venv\Scripts\python.exe -m shot_predict.autopilot.cli add-match --source "C:\path\full_match.mp4" --match-id usopen_2026_r4 --near-player Alcaraz --far-player Paul
# 2. Analyze every pending clip with several worker processes. Safe to stop (Ctrl+C) and rerun;
#    it keeps working while you validate, and finished clips show up as soon as you reopen Validate.
.\.venv\Scripts\python.exe -m shot_predict.autopilot.cli batch --workers 3
.\.venv\Scripts\python.exe -m shot_predict.autopilot.cli status
# 3. Start the app as usual. New matches import themselves; press "Validate" (or V).
```

**One-time setup per camera.** The analysis finds the court by matching each frame to a reference
frame whose corners you clicked once. For a new tournament or camera, open any clip in the
app (it opens in court setup), pause on a sharp wide view of the whole court, click the four
singles corners, and press **Register this camera for all clips**. Corners that do not line up
with the court lines are refused on the spot. After that, every clip from that camera in any
match is handled automatically; `status` lists clips that need a camera or show no court view.
Other cameras (for example the low behind-the-baseline view) are not supported and are skipped.

**Keys:** `Enter`/`Y` correct, `X`/`N` wrong, `Q` unsure, `E` fix it in the detailed review
flow, `R` replay, `Space` pause, `Z` undo, `A` audits, `V` leave.
Decisions are saved in the background in batches, so the loop never waits for the app.

**What a card shows.** The loop starts just before contact and runs to just after the bounce
at half speed with the hitter box flashing at contact, the ball's flight drawn as it goes, and
the proposed landing zone (purple) once it bounces. You confirm that the ball landed in that zone
and that the highlighted player hit it. A confirmed card is saved exactly as proposed
(`source: auto_confirmed`, nothing edited, with your review time); the label always comes from
the analysis on disk, never from the browser.

**How good are the cards? (measured, small samples)** Only proposals that carry a landing
zone become cards, because without one there is nothing to confirm. On the labeled Turin
rallies such proposals were fully right (hitter, zone, and contact within 250 ms) for 22 of 27,
81%; the other 32 proposals had no zone and are never shown. Among the right ones, 6 of 22
had contact timing more than 100 ms from yours, which is label noise you accept by confirming.
On your first US Open labels, the 5 zone-bearing proposals with correct timing were 5/5 right.
Nothing finer than "has a zone" predicted correctness (81-86% at every confidence
threshold), so there is no strictness knob; expect roughly one wrong card in five, and these
numbers are from one match, so watch your own accept rate on the stats line.

**Audit cards.** About one card in twelve is a deliberate decoy: the real proposal with a
clearly wrong zone drawn on it. Marking it Correct is counted ("audits 3/4 caught"), no label
is saved for it, and the real card follows immediately. It measures how carefully you are
validating at speed. Turn it off with `A`.

**Limits to keep in mind.**
- *Selection bias.* Only shots whose bounce the tracker could see ever become cards, so a
  dataset made only of confirmed cards over-represents clean, trackable shots. Keep manual
  or fixed-camera labels in the mix and compare with
  `python -m shot_predict.train --label-sources manual auto_corrected auto_confirmed`
  against runs trained on one source at a time.
- Serves are guessed from "first shot of a rally"; check the type chip.
- Bounce frames are the engine's estimate; contact frames are the (corrected) engine's.
  Use `E` when timing matters.
- **Analysis speed (measured on this CPU-only laptop, 1080p60 clips of about 5 minutes).**
  One clip alone took about 10 minutes before the player-detection step was made time-based;
  that step now takes about 94 s instead of about 220 s per clip with identical proposals on
  the clip I compared (42 vs 42, all 39 shared proposals identical). Three workers in
  parallel finished three clips in 15, 17 and 19 minutes of wall time while I was also running
  tests and browser checks, about 10 clips an hour. At that rate a 3-hour match (about 36
  clips) is several hours of unattended analysis and tens of matches are days, so run it
  overnight; each worker used 0.6-0.9 GB, so `--workers 5` is possible if you leave the machine
  alone. The GPU is not used (this install of PyTorch is CPU-only) and would only speed up the
  player detector, not the ball tracker, which is the other half of the time.

## Train

Label **at least two distinct matches** before training:

```powershell
.\.venv\Scripts\python.exe -m shot_predict.train --pretrained --freeze-backbone
```

This downloads public ImageNet ResNet18 weights, freezes the CNN initially,
and trains a temporal GRU/head. Without `--pretrained`, the full network is
trained from scratch without downloading weights. To fine-tune the CNN,
omit `--freeze-backbone`. Each new run needs a new output directory:

```powershell
.\.venv\Scripts\python.exe -m shot_predict.train --pretrained --output runs\finetune --epochs 20 --validation-match held-out-match-id
```

This starts a fresh run; it does not resume the previous checkpoint.
Defaults: 8 sampled frames over 0.6 seconds, 224px images, and prediction
leads of 100, 200, and 300ms before contact. Far-hitter frames are rotated
180 degrees to normalize hitting direction; zone labels remain defined
from the receiver's perspective. Resizing the full frame is a simple
baseline, not a ball-tracking or pose-aware architecture.

Contact and post-contact frames are **always excluded from model input**,
even at `--lead-ms 0`. Bounce positions are labels only. Examples without
enough pre-contact footage produce explicit errors rather than padded or
silently skipped samples. Label contacts at least one second into a clip
for the default settings.

Splits are grouped by match ID, never by neighboring frames. The default
holds out roughly 20% of labeled matches (at least one). Repeat
`--validation-match` to select multiple held-out matches explicitly.
The report warns about missing training classes, missing player names, and
player overlap. A match holdout alone does not prove unseen-player ability.

Outputs:

- `runs\baseline\best.pt`: checkpoint chosen by mean validation negative
  log likelihood across lead times.
- `runs\baseline\metrics.json`: top-1/top-2 accuracy, confusion matrices,
  accuracy by near/far hitter and shot type, balanced accuracy over present
  classes, and the training-majority baseline, all per lead time.

Compare against the majority baseline and inspect all six classes, both
ends, and serves separately. Keep an additional untouched test set when
you start tuning models. Softmax probabilities are **not calibrated
confidence**; reliability calibration is future work.

For a modest CPU, reduce `--image-size 128 --samples 4 --batch-size 2`.
`--device auto` selects CUDA when available, otherwise CPU. For NVIDIA
acceleration, install the appropriate PyTorch CUDA distribution using the
official PyTorch installation instructions; the default pip environment
may be CPU-only.

### Exploratory training with just one match

For a quick sanity check before collecting a second match, explicitly hold out
one or more **whole source clips** using their video IDs from the project:

```powershell
.\.venv\Scripts\python.exe -m shot_predict.train --pretrained --freeze-backbone --cache-features --augmentation-copies 3 --exploratory-validation-video HELD_OUT_VIDEO_ID --image-size 160 --samples 4 --hidden 64 --lead-ms 200 --learning-rate 0.001 --epochs 30 --output runs\exploratory
```

This opt-in mode is labeled `exploratory_same_match_video_holdout` in the
checkpoint and metrics. It does **not** replace the default match-separated
evaluation, and its scores are not evidence of performance on new matches.
Do not use overlapping footage or clips that split the same rally across the
training/validation boundary.

The pretrained image encoder stays frozen; only the temporal GRU and classifier
learn from your labels. Original training sequences plus the requested number
of augmented copies are encoded once and cached in memory, making CPU training
faster. Each copy uses mild brightness/contrast/saturation changes, rotation,
translation, and scaling consistently across its frames. Horizontal mirrors
swap ad/deuce labels while preserving middle and depth. Validation is never
augmented. Copies are not independent new shots, and augmentation cannot
replace data from different players and matches.

For imbalanced zones, `--class-weighting inverse-sqrt` gives less common
classes more weight in the training loss. Weights are computed from the actual
training examples, including any augmented copies; validation is unchanged.
The default is `--class-weighting none`. Weighting cannot teach a class with
no examples, and should be compared on held-out data rather than assumed better.

Reports include the split IDs, original/augmented class counts, best epoch,
best validation metrics, and complete epoch history. Checkpoint selection uses
validation loss, so the validation scores are not an untouched test result.

### Inspect inputs and prediction heatmaps

For a run containing `best.pt`, `metrics.json`, and `project_snapshot.json`:

```powershell
.\.venv\Scripts\python.exe -m shot_predict.explain --run runs\exploratory-121
```

This writes a new `explanations` folder without changing the model or labels.
It contains original training-input examples, illustrative augmentations, an
overview of held-out Grad-CAM maps, all sampled frames and heatmaps for each
held-out shot, and a manifest with exact source frame indices/probabilities.
Augmentation illustrations use the current policy, not the exact cached copies
from the earlier training run. Use `--limit 4` for a smaller preview or
`--output` to choose another new folder; existing output folders are rejected.

The original full-frame exploratory model receives four broadcast frames resized to 160x160,
at approximately 800, 600, 400, and 200 ms before contact. Far-hitter sequences
are rotated 180 degrees, and RGB values use ImageNet normalization. There is
no player/ball crop or masking of spectators, logos, and scoreboards.

Grad-CAM explains the **predicted class logit through both the CNN and GRU**.
The frozen CNN is differentiated with respect to the input, without updating
its weights. At 160x160 the last convolutional grid is only 5x5. Positive maps
share one normalization across the four frames of each sequence, but scales
cannot be compared across shots. A missing overlay means no positive signal.
These coarse maps are not calibrated attention, ball tracking, or proof of a
causal explanation. Held-out validation was used for checkpoint selection; it
is not an untouched test set.

### Optional YOLO hitter crops

Install the local detector support (the headless package avoids overlapping
OpenCV GUI/headless installations):

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[players]"
```

YOLO11n downloads its public pretrained weights on first use to
`data\models\yolo11n.pt`. Video inference stays local; frames are not uploaded.
Telemetry and experiment integrations are disabled, with detector settings
isolated under `data\.ultralytics`. Ultralytics uses
[AGPL-3.0 or an enterprise license](https://www.ultralytics.com/license);
review those terms before distributing or using this dependency in a
proprietary product.

Preprocess an existing run's project snapshot and exact training/validation
shot IDs into a **new** directory:

```powershell
.\.venv\Scripts\python.exe -m shot_predict.players --run runs\exploratory-121 --output runs\player-crop-121-centered
```

Detection uses only the sampled pre-contact frames. Court calibration and the
annotated near/far hitter select a person, with continuity within the sequence;
landing coordinates and destination labels never select the crop. Each square
follows the player, with one stable size across the sequence: 1.6 times the
largest person-box dimension. Out-of-image regions use ImageNet-mean padding.
Both hitters remain **upright**, unlike the full-frame far-hitter rotation.
This is person detection/cropping, not background removal or racket tracking;
the surrounding court, graphics, and sometimes the ball remain visible.

Inspect `all_player_crops.jpg` and the `crop_review_*.jpg` source-box audits
before training. Missing hitters are recorded explicitly; unresolved failures
block training rather than skipping shots or substituting full-court inputs.
Cache identity includes source path/size/modification time, frame indices,
court calibration, hitter, image size, crop version/settings, and detector
SHA-256. Changes require rebuilding the cache. Retain the source videos.

For the 121-shot experiment, reproduce the original model and split settings:

```powershell
.\.venv\Scripts\python.exe -m shot_predict.train --project runs\player-crop-121-centered\project_snapshot.json --output runs\player-crop-121-centered --player-cache runs\player-crop-121-centered\player_cache --pretrained --freeze-backbone --cache-features --augmentation-copies 3 --exploratory-validation-video cb30ef3c8ad23175 --image-size 160 --samples 4 --hidden 64 --lead-ms 200 --learning-rate 0.001 --epochs 30 --batch-size 8 --threads 4 --device cpu
.\.venv\Scripts\python.exe -m shot_predict.explain --run runs\player-crop-121-centered
```

This trained on the same 93 shots plus 279 augmented copies and validated on
the same 28 shots. Checkpoints were chosen by minimum validation NLL:

| Input | Best epoch | Top-1 | Top-2 |
| --- | --- | --- | --- |
| Full frame | 13 | 5/28 (17.9%) | 15/28 (53.6%) |
| Upright player crop | 1 | 8/28 (28.6%) | 16/28 (57.1%) |
| Training-frequency baseline | - | 7/28 (25.0%) | 16/28 (57.1%) |

Three additional correct predictions on one same-match validation clip are
not convincing evidence of generalization. The crop model still misses entire
classes, and its coarse Grad-CAM maps still highlight some background. Cropping
also removes global player position/movement and opponent context; upright
orientation is another difference from the full-frame baseline. More varied
matches, player-focused fine-tuning or pose features, and an untouched test set
are still needed to evaluate this direction.

## Export a prediction video

Import/calibrate the input in the app. The app's "Training and prediction
commands" section supplies its video ID and a ready-to-copy export command:

```powershell
.\.venv\Scripts\python.exe -m shot_predict.predict --video-id YOUR_VIDEO_ID --hitter both --output runs\prediction.mp4
```

The exporter consumes only the current/past frames, not annotation contact
or bounce times. It produces an MP4 with continuously updating near/far
zone arrows and a CSV of six probabilities per update per hitter.
It retains source frame rate, resolution, and frame count, but **not audio**.
Existing output files are not overwritten.

`--hitter near` or `--hitter far` restricts forecasts to one selected player.
`--stride 3` updates every third frame; `--smoothing 0.5` smooths updates
causally; `--threshold 0.45` hides arrows below the highest raw probability.

For a crop checkpoint, add
`--checkpoint runs\player-crop-121-centered\best.pt`. The exporter automatically
uses the matching detector and player-centered preprocessing for either/both
hitters. If a hitter is missing, its stale forecast and smoothing state are
cleared; the video displays an unavailable message. Crop-mode CSVs include a
`status` column and blank probabilities for those updates. Original full-frame
checkpoint behavior and CSV columns are unchanged. This detector runs on CPU;
no real-time latency claim has been established.

### Honest first-version limitations

- No automatic racket contact, turn detection, persistent player tracking, or replay
  filtering yet. In `both` mode, both estimates are drawn, including times
  when the respective player is not about to hit. Predictions outside
  pre-contact situations are out of the training distribution.
- Tracers start from **fixed baseline-center reference points**, not tracked
  player locations, and point at a **zone center**, not an exact landing
  coordinate or physical ball-flight trajectory.
- No useful accuracy can be claimed without real labeled data. A successful
  synthetic test only verifies the software pipeline.
- This is prerecorded processing, not a real-time performance claim.
  Contact gating, persistent tracking, and latency benchmarking remain future
  work; optional YOLO crops do not establish live performance.
- Serves, court conditions, broadcast graphics, camera differences, and
  unfamiliar players can all shift the input distribution. Generalization
  to every professional player is not guaranteed.

### Recorded rally visualization

The [21.8-second rally demo](runs/player-crop-367/rally-demo/tennis_prediction_and_observed_ball.mp4)
shows YOLO player boxes, magenta dashed arrows to predicted zone centers, and
yellow observed-ball trails. Its run-specific
[renderer](runs/player-crop-367/rally-demo/render_rally_demo.py) has separate
`analyze` and `render` stages and refuses to overwrite existing outputs.

Unlike the continuous exporter above, this replay uses labeled hitter/contact
timing and saved held-out predictions from the 367-shot experiment. Arrow
origins approximate the hitter's body position. All top-1 forecasts are shown,
including scores below the exporter's unchanged 0.45 threshold; these are not
calibrated confidence values or exact flight predictions.

The ball detector uses color/motion, **not YOLO or ground-truth ball labels**.
It found candidate ball positions on 306 of 545 frames; missing observations
break the yellow trail rather than being interpolated. Player boxes are
interpolated between successful YOLO samples, with five near-player frames
unavailable. This is an annotation-timed prerecorded visualization, not an
autonomous live tracker or a claim of accuracy on other courts.

### Player-position experiment

`runs/player-crop-367/position-experiment/` tests whether both players'
court positions predict the zone, using only the cached pre-contact YOLO
person boxes (no new labels). `position_features.py` maps each player's feet to
court metres, rotated so near and far hitters share one receiver-relative frame.
`position_experiment.py` selects models only by leave-one-clip-out
cross-validation on the 316 train/validation shots, then touches the 47-shot test
split once. `position_followup.py` separates serves from rally shots.

Findings (same match, so not evidence for other courts or players): positions
beat the majority baseline mainly because they reveal serves, which land short.
For rally shots they mostly inform lateral direction, and the gain over the
common-zone baseline is modest (101/227 vs 86/227 in cross-validation, with a
bootstrap interval that includes zero). The 25/47 held-out test result is higher
than the cross-validated rate and has only 46 scored shots. Fusing the CNN
changed top-1 by zero shots. The features use labeled contact timing, so a live
system would also need a contact/about-to-hit detector.

`runs/player-crop-367/rally-demo/render_position_demo.py` renders the saved
held-out position-model forecasts onto the same 15-shot rally excerpt (player
boxes, model-input foot points, predicted zone, and the labeled actual zone
revealed only after the bounce) with a running tally against the old CNN and a
common-zone baseline. It reuses the earlier `analysis.json` and, like the other
replay, relies on labeled contact timing and is not live inference.

### Combined Turin + US Open model (506 labels)

`runs\combined-506\` repeats the player-crop training with the US Open labels you validated
(367 manual Turin labels, 3 more confirmed Turin shots, 136 US Open shots: 106 confirmed
and 30 corrected). Everything is reproducible from the scripts in that folder:

```powershell
.\.venv\Scripts\python.exe runs\combined-506\make_split.py                       # whole-clip split + manifests
.\.venv\Scripts\python.exe -u -m shot_predict.players --run runs\combined-506\train-source --output runs\combined-506\train-crops --threads 4
.\.venv\Scripts\python.exe -u -m shot_predict.players --run runs\combined-506\test-source --output runs\combined-506\test-crops --threads 4
.\.venv\Scripts\python.exe runs\combined-506\build_selected_cache.py             # drops shots with no detected hitter, recorded
# one run per candidate (add --class-weighting inverse-sqrt for the weighted one); repeat --validation-video per validation clip
.\.venv\Scripts\python.exe -m shot_predict.train --project runs\combined-506\training_project_selected.json --player-cache runs\combined-506\selected-player-cache --pretrained --freeze-backbone --cache-features --augmentation-copies 3 --image-size 160 --samples 4 --hidden 64 --lead-ms 200 --learning-rate 0.001 --epochs 30 --batch-size 8 --threads 4 --device cpu --seed 42 --validation-video <clip id> --output runs\combined-506\candidate-unweighted
.\.venv\Scripts\python.exe runs\combined-506\select_candidate.py                 # lowest validation NLL; test unused
.\.venv\Scripts\python.exe runs\combined-506\evaluate_test.py                    # test_metrics.json, test_predictions.csv
.\.venv\Scripts\python.exe runs\combined-506\explain_test_samples.py             # Grad-CAM sheets + player-evidence statistic
.\.venv\Scripts\python.exe runs\combined-506\rally-demo\render_usopen_demo.py    # held-out US Open replay video
.\.venv\Scripts\python.exe runs\combined-506\position_model.py                   # position model on the same split
```

`--validation-video` (new) holds out whole clips from any matches, unlike
`--exploratory-validation-video`, which requires one match.

**Split.** Whole clips only. Turin keeps the earlier partitions (validation clips 005 and 012,
test clips 015-017). US Open: clips 000-003 train, 004 validation, and the two latest labeled
clips (005, 006) are the test. That is 351 training, 72 validation and 77 test shots, every
boundary between partitions at least 34 s apart in the original recording, test never used for
training or for choosing the checkpoint. Alcaraz plays in both matches, so no split is an
unseen-player evaluation.

**Held-out test (76 shots with crops; 95% intervals are wide).**

| | Turin test (46) | US Open test (30) | All (76) |
| --- | --- | --- | --- |
| Combined model | 18 (39.1%) | 13 (43.3%) | **31 (40.8%, CI 30-52%)** |
| Previous Turin-only model | 16 (34.8%) | 12 (40.0%) | 28 (36.8%) |
| Always guess the training-majority zone | 15 (32.6%) | 7 (23.3%) | 22 (28.9%) |

The previous Turin-only model scored 43 of 134 (32.1%) on **every** US Open shot, which it never saw,
against 28.4% for the majority guess: that is the honest new-court number, and it is close to guessing.
The combined model's US Open test shares the match with its training clips, so it is not comparable.

**What the 40.8% actually is.** Per zone, the model predicts `ad_deep` for 47 of 76 shots and gets
0 of 28 deuce-side shots right (`deuce_deep` 0/19, `deuce_short` 0/9). Its edge comes from
recognizing **serves**: 13 of 14 serves are predicted short (exact box 7/14). On rally shots it is
at baseline: 24/62 against 22/62 for always guessing `ad_deep`. Raw probabilities stay between
0.24 and 0.51 (only 2 of 76 reach the exporter's 0.45 default) and are uncalibrated.
More data of the same kind will not obviously change this: the pre-contact body pose 0.2-0.8 s
before contact says little about a groundstroke's destination, which matches the earlier result.

**Grad-CAM** (`test_heatmaps_turin.jpg`, `test_heatmaps_usopen.jpg`): examples are the first test
shot per true zone, chosen by rule, not by outcome. Over all 76 test shots, 59% of the positive
Grad-CAM evidence falls on the central 3x3 of the 5x5 map, where the player is (uniform
would be 36%): Turin 62%, US Open 55%, far hitters 65%, near hitters 53%, correct predictions 63%
versus incorrect 56% (`test_heatmaps.json`). So with crops the model does look at the player, as
you wanted. The maps are coarse and do not show that the player's pose predicts direction.
Some US Open serve inputs contain a broadcast camera cut inside the 0.6 s input window.

**Prediction video** (`runs\combined-506\rally-demo\tennis_prediction_usopen_heldout.mp4`): 14
labeled shots from three held-out US Open rallies, with the court tracked per frame as the camera
pans. Forecast zone in magenta (not ball flight), the tracker's ball in yellow, the true zone in
green only after the bounce, with a running tally. Excerpts were fixed before any model output
existed; it is a prerecorded, label-timed replay, not live inference, and 6 of 14 exact (8 on
side, 12 on depth) is an illustration, not an accuracy estimate.

**Position model on the same split** (`position_model.py`): hitter and opponent court positions
from the cached detections, regularization by leave-one-clip-out CV on train+validation clips only.
On the 75 test shots with features: positions alone 35 (46.7%, CI 36-58%), positions plus a
serve flag 32 (42.7%; the variant with the better CV log loss), CNN 31 (41.3%), majority guess 22
(29.3%). It gets 15 of 19 `deuce_deep` shots where the CNN gets none, but is much better on Turin
(24/46) than on US Open (11/29), possibly because the moving camera adds apparent motion to the
features. These differences are within the noise of 75 shots.

## Validate

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Tests cover both-end geometry, label persistence, leakage-safe sampling,
match splits, model output shape, local app startup, and a synthetic
training-to-video-export run. Synthetic runs download no model weights.
