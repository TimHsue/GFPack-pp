"""Regression checks for bounded generation and valid geometric partitions."""

import importlib.util
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("MPLBACKEND", "Agg")
import numpy as np
from shapely.affinity import affine_transform
from shapely.geometry import Polygon
from shapely.ops import unary_union

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "genPuzzle.py"
spec = importlib.util.spec_from_file_location("gen_puzzle", SCRIPT)
puzzle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(puzzle)


class PuzzleTests(unittest.TestCase):
    def run_cli(self, directory, *args):
        return subprocess.run(
            [sys.executable, str(SCRIPT), *map(str, args)], cwd=directory,
            capture_output=True, text=True, timeout=20,
        )

    def assert_partition(self, boundary, cut=15, rotation=True, seed=3407):
        puzzle.random.seed(seed)
        np.random.seed(seed)
        source = Polygon([(p.x, p.y) for p in boundary])
        vertices, targets, _ = puzzle.genDataPoly(
            [boundary], cut=cut, rotation=rotation, resample_num=0)
        self.assertEqual(len(vertices), cut + 1)
        placed = []
        for ring, pose in zip(vertices, targets):
            x, y = pose[:2]
            c, s = pose[2:] if rotation else (1, 0)
            shape = affine_transform(Polygon(ring), [c, -s, s, c, x, y])
            self.assertTrue(shape.is_valid)
            self.assertGreater(shape.area, 0)
            placed.append(shape)
        union = unary_union(placed)
        tolerance = source.area * 1e-8
        self.assertLess(union.symmetric_difference(source).area, tolerance)
        self.assertLess(abs(sum(p.area for p in placed) - union.area), tolerance)

    def test_rectangle_both_windings_and_rotation_modes(self):
        coordinates = [(0, 0), (2000, 0), (2000, 2000), (0, 2000)]
        for reverse in (False, True):
            for rotation in (False, True):
                for seed in (0, 1, 3407):
                    with self.subTest(reverse=reverse, rotation=rotation, seed=seed):
                        ring = coordinates[::-1] if reverse else coordinates
                        self.assert_partition([puzzle.TPoint(*p) for p in ring],
                                              rotation=rotation, seed=seed)

    def test_concave_boundary(self):
        boundary = puzzle.scaleTo(
            [(0, 0), (4, 0), (4, 1), (1, 1), (1, 4), (0, 4), (0, 0)], 2000)
        for seed in range(10):
            with self.subTest(seed=seed):
                self.assert_partition(boundary, seed=seed)

    def test_path_is_bounded_when_no_exit_is_found(self):
        boundary = puzzle.scaleTo([(0, 0), (1, 0), (1, 1), (0, 1)], 2000)
        with patch.object(puzzle, 'minCrossPoint', return_value=(None, None)) as intersections:
            self.assertEqual(puzzle.cutCurrent(boundary, max_path_steps=3), (None, None))
            self.assertEqual(intersections.call_count, 3)

    def test_failed_cuts_are_bounded(self):
        boundary = puzzle.scaleTo([(0, 0), (1, 0), (1, 1), (0, 1)], 2000)
        with patch.object(puzzle, 'cutCurrent', return_value=(None, None)) as cutter:
            with self.assertRaisesRegex(RuntimeError, 'after 3 attempts'):
                puzzle.genDataPoly([boundary], cut=1, max_cut_attempts=3)
            self.assertEqual(cutter.call_count, 3)

    def test_invalid_boundaries(self):
        for ring in ([(0, 0), (1, 1), (0, 1), (1, 0)],
                     [(0, 0), (1, 0), (2, 0)],
                     [(0, 0), (1, float('nan')), (0, 1)]):
            with self.subTest(ring=ring), self.assertRaises(ValueError):
                puzzle.scaleTo(ring, 2000)

    def test_rectangle_cli_without_boundary_file_and_small_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_cli(directory, '--rect', 1, '--cut', 3,
                                  '--datasize', 2, '--output', 'nested/dataset.pkl',
                                  '--rotation', 0)
            self.assertEqual(result.returncode, 0, result.stderr)
            with (Path(directory) / 'nested/dataset.pkl').open('rb') as file:
                vertices, poses, ids = (pickle.load(file) for _ in range(3))
                self.assertEqual(file.read(), b'')
            self.assertEqual(len(vertices), 2)
            self.assertEqual([len(p) for p in vertices], [4, 4])
            self.assertEqual(len(poses[0][0]), 2)
            self.assertEqual(ids, [-1, -1])
            self.assertEqual(len(list((Path(directory) / 'puzzle_vis').glob('*.png'))), 2)

    def test_json_and_pickle_boundaries_and_seed_reproducibility(self):
        boundaries = [[(0, 0), (4, 0), (4, 1), (1, 1), (1, 4), (0, 4)]]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'boundary.json').write_text(json.dumps(boundaries))
            with (root / 'boundary.pkl').open('wb') as file:
                pickle.dump(boundaries, file)
            for extension in ('json', 'pkl'):
                result = self.run_cli(directory, '--boundary', f'boundary.{extension}',
                                      '--cut', 5, '--datasize', 1, '--vis_count', 0,
                                      '--output', f'dataset.{extension}.pkl')
                self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual((root / 'dataset.json.pkl').read_bytes(),
                             (root / 'dataset.pkl.pkl').read_bytes())

    def test_generation_failure_preserves_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'dataset.pkl'
            output.write_bytes(b'previous dataset')
            result = self.run_cli(directory, '--rect', 1, '--cut', 1,
                                  '--datasize', 1, '--step', 1,
                                  '--max_path_steps', 1, '--max_cut_attempts', 1,
                                  '--vis_count', 0)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Failed to generate sample 0', result.stderr)
            self.assertEqual(output.read_bytes(), b'previous dataset')

    def test_invalid_cli_parameters(self):
        with tempfile.TemporaryDirectory() as directory:
            for option, value in (('--step', 0), ('--cut', -1),
                                  ('--datasize', 0), ('--resample_num', 2)):
                with self.subTest(option=option):
                    result = self.run_cli(directory, '--rect', 1, option, value)
                    self.assertEqual(result.returncode, 2)


if __name__ == '__main__':
    unittest.main()
