"""Unit tests for the standalone v2 pose feature and grouping pipeline."""

from __future__ import annotations

import unittest

import numpy as np

from pose_features_v2 import FEATURE_SIZE, extract_temporal_features, normalize_landmarks
from train_pose_v2 import ClipRecord, split_grouped_indices


class PoseFeatureTests(unittest.TestCase):
    def test_normalization_removes_translation_and_uniform_scale(self) -> None:
        rng = np.random.default_rng(4)
        clip = rng.normal(size=(6, 33, 3))
        shifted_scaled = clip * 3.5 + np.asarray([12.0, -4.0, 7.0])
        np.testing.assert_allclose(
            extract_temporal_features(clip),
            extract_temporal_features(shifted_scaled),
            rtol=1e-10,
            atol=1e-10,
        )

    def test_feature_shape_is_fixed_for_single_frame_and_clip(self) -> None:
        clip = np.zeros((4, 33, 3), dtype=np.float64)
        self.assertEqual(extract_temporal_features(clip).shape, (FEATURE_SIZE,))
        self.assertEqual(normalize_landmarks(clip[:1]).shape, (1, 33, 3))


class GroupSplitTests(unittest.TestCase):
    def test_grouped_splits_are_pairwise_disjoint_and_reproducible(self) -> None:
        records = [
            ClipRecord(path=None, label=f"label_{index % 2}", group=f"source_{index // 2}")
            for index in range(12)
        ]
        first = split_grouped_indices(records, seed=19)
        second = split_grouped_indices(records, seed=19)
        self.assertEqual(set(first), {"train", "validation", "test"})
        self.assertEqual({tuple(value) for value in first.values()}, {tuple(value) for value in second.values()})
        self.assertTrue(set(first["train"]).isdisjoint(first["validation"]))
        self.assertTrue(set(first["train"]).isdisjoint(first["test"]))
        self.assertTrue(set(first["validation"]).isdisjoint(first["test"]))
        assigned = np.concatenate(list(first.values()))
        self.assertEqual(sorted(assigned.tolist()), list(range(len(records))))
        for name, indices in first.items():
            groups = {records[index].group for index in indices.tolist()}
            self.assertGreater(len(groups), 0, name)

    def test_rejects_fewer_than_three_source_groups(self) -> None:
        records = [
            ClipRecord(path=None, label="squat", group="source_a"),
            ClipRecord(path=None, label="squat", group="source_b"),
        ]
        with self.assertRaisesRegex(ValueError, "at least 3 distinct source groups"):
            split_grouped_indices(records)


if __name__ == "__main__":
    unittest.main()
