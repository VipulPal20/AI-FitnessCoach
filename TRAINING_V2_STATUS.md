# Pose v2 training status

## Actual completed run

Downloaded the pose CSV and manifest (not videos) from version 4 of:
https://www.kaggle.com/datasets/bangaji/attentiveskel3d-weighttraining-dataset-per-frame

Imported 487 clips, each containing 64 MediaPipe poses. Trained a separate
DecisionTreeClassifier on normalized temporal features for bench_press,
deadlift, and squat. This is an exploratory movement classifier, NOT the
completed hierarchical pose/equipment system.

Artifacts: `models/attentive_research_tree.joblib` and
`models/attentive_research_metrics.json`.

The published CSV does not link each clip to its original recording/person.
Therefore this evaluation separates clips only: related repetitions or people
may appear in different splits. Do not treat its accuracy as unseen-video
accuracy, or as evidence that bench press vs shoulder press is fixed.

Kaggle lists MIT, but the data includes derivatives of Hasyim's non-commercial
dataset. This candidate is for research; commercial clearance is unresolved.
No third-party Python or pickle files were downloaded or executed.

## Pipeline

- `pose_features_v2.py`: shared normalized temporal pose features.
- `extract_pose_v2.py`: local-video manifest to pose clips; records provenance.
- `import_attentive_pose.py`: imports the published pose CSV without video downloads.
- `train_pose_v2.py`: separate fitted tree and explicit missing-class metrics.
- `test_pose_v2.py`: normalization and split-isolation tests.

## Still missing

Shoulder press, push-up, bicep curl, pull-up, row, lunge, jumping jack, and
unknown/non-exercise are not in this candidate. Equipment detection, online
integration, expanded rep counters, and held-out raw-video evaluation remain
unfinished. Production still uses the old artifact. Its pull-up-to-curl and
push-up-to-shoulder-press label mappings are invalid; previous reported accuracy
did not validate those displayed exercises.

No videos were downloaded by this run, so no new training videos require
deletion. Temporary CSV/pose data are outside the repository. Existing user
datasets and earlier test videos have not been deleted.
