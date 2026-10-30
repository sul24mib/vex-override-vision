# VEX U Override: game-object detection (WPI MQP)

Camera perception for our VEX U robot: find pins, cups, goals, toggles, loaders, robots and people on the
**Override** field, from the robot's 4-camera rig (Arducam Camarray → Raspberry Pi 5 → NVIDIA Jetson AGX Orin).

Status (Oct 2026): detector **v1** is trained on automatic labels; the team is hand-checking 201 pictures to
measure v1 honestly and train **v2**. SAM 3 was tested as an automatic labeller.

## Pipeline
```
field recording (4 cameras, 20 fps)
  -> tools/colour_labels.py     draft labels from colour + a SAM 2 floor mask (no training)
  -> tools/build_dataset.py     split by TIME into train / val / test (70 / 10 / 20 %)
  -> tools/yolo_speed.py        pick a model size that is fast enough on the Jetson
  -> tools/train_yolo.py        train YOLO11 + score on the test split
  -> tools/eval_v1.sh           re-test v1, then run the SAM 3 test
  -> tools/make_label_batches.py  choose 201 pictures for people to correct (4 batches)
  -> labelling/VEX_labelling.ipynb  Colab notebook the team uses to correct the boxes
```
`tools/run_labels_train.sh` runs the first five steps in one go on the Jetson.

## Files
| File | What it does |
|---|---|
| `tools/colour_labels.py` | Draft YOLO labels: HSV colour ranges for pin parts, SAM 2 floor mask to drop things outside the field |
| `tools/build_dataset.py` | Builds the YOLO dataset, split by time so near-identical frames don't leak into the test set |
| `tools/yolo_speed.py` | Speed test of YOLO11 n / s / m, PyTorch vs TensorRT, on the Jetson |
| `tools/train_yolo.py` | Trains the chosen model and scores it on the test split |
| `tools/detect_photos.py` | Earlier test: YOLO-World (finds objects named in words, no training) on a folder of pictures |
| `tools/sam3_test.py` + `run_sam3_test.sh` | SAM 3 with text prompts ("yellow pin", "robot", "goal", ...): boxes, counts and time per picture |
| `tools/eval_v1.sh` | Re-evaluates v1 on the test split, then runs the SAM 3 test |
| `tools/make_label_batches.py` | Picks 141 test pictures (evenly over time) + 60 hard training pictures (where v1 is least sure), pre-draws v1's boxes, splits into 4 batches |
| `labelling/make_notebook.py` | Writes `VEX_labelling.ipynb`: a Colab notebook that reads a batch from a shared Google Drive, lets a person fix the boxes, and saves YOLO labels back after every picture |
| `models/yolo11m_override_v1.pt` | The v1 detector (YOLO11m, 5 classes) |
| `results/` | v1 scores and training curves, SAM 3 test pictures, the evaluation log |

## Results so far

### Speed on the Jetson AGX Orin (sets of 4 camera pictures per second)
| Model | Image size | PyTorch | TensorRT FP16 |
|---|---|---|---|
| YOLO11n | 640 | 30.6 | 40.0 |
| YOLO11s | 640 | 22.6 | 26.7 |
| YOLO11s | 416 | | 38.6 |

### Detector v1
YOLO11m, 640 px, trained 60 epochs on the automatic (colour + SAM 2) labels; about 10 ms per picture on the Jetson.

| Test split | mAP50 | mAP50-95 | Precision | Recall |
|---|---|---|---|---|
| all classes | 0.650 | 0.522 | 0.542 | 0.707 |

Per class mAP50: yellow part 0.775, blue part 0.512, red part 0.656, red goal 0.738, blue goal 0.567.

**Important limit:** these scores are measured against the same kind of *automatic* labels v1 was trained on, so
they are not its real accuracy. Looking at the pictures, v1 misses pins in shadow, stacked pins, cups and robots,
and its "red goal" / "blue goal" boxes often land on red or blue pins close to the camera. The hand-checked
pictures (in progress) will give the honest score.

### SAM 3 as an automatic labeller
8 field pictures, text prompts, about 0.8 s per picture on the Jetson.
- Good: people (0.95+ confidence), many loose pins.
- Bad: calls pin stacks and goals "robot", puts "clear plastic cup" on many things that are not cups, found 0 goals.
- Conclusion: SAM 3 does not know VEX game pieces well enough to replace people; useful for people and as a rough first draft.

## Labelling rules (v2 class list, from the Override game manual v2.1)
15 classes: `yellow part, blue part, red part, red goal, blue goal, robot, person, pin, neutral goal, cup,
red toggle, blue toggle, yellow toggle, red loader, blue loader`.
- Box everything you can see. Each **pin** gets a pin box plus one box per visible coloured **half**
  (each half scores on its own; the pin box tells the robot which halves belong together, for picking).
- Stacks: goal, each cup, each pin and its visible halves get separate boxes; nothing that is hidden.
- **Toggle**: one box around the bar; class = colour of its top face (yellow over red = yellow, blue over yellow = blue, red over blue = red).
- Pieces inside a **loader** are boxed like any other piece, plus one box around the loader.

## Running
On the Jetson (Ultralytics must not install packages: it once upgraded numpy and broke other tools):
```bash
export YOLO_AUTOINSTALL=false
bash tools/run_labels_train.sh                       # labels -> dataset -> training -> test
yolo predict model=models/yolo11m_override_v1.pt source=<pictures> conf=0.25   # run v1 on pictures
python tools/sam3_test.py <pictures> --out results/sam3_test   # needs access to the gated SAM 3 weights
```
Model weights for SAM 2 / SAM 3 / YOLO base models are downloaded by Ultralytics and are not in this repository.
