"""CPU checks for token coverage, tail boundaries, and existing workflow settings."""
import importlib.util
from pathlib import Path
import types
import unittest
from unittest.mock import patch

import torch

SPEC = importlib.util.spec_from_file_location("h3_ffn_under_test", Path(__file__).resolve().parents[1] / "ffn.py")
ffn = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ffn)


class Model:
    def __init__(self, mlp):
        self.diffusion = types.SimpleNamespace(blocks=[types.SimpleNamespace(mlp=mlp)])
        self.patches = {}

    def clone(self):
        return Model(self.diffusion.blocks[0].mlp)

    def get_model_object(self, name):
        return self.diffusion

    def add_object_patch(self, name, value):
        self.patches[name] = value


class FeedForwardTests(unittest.TestCase):
    def run_partition(self, rows, chunks):
        sizes = []
        def fc1(x):
            sizes.append(len(x))
            return x
        original = Model(types.SimpleNamespace(fc1=fc1, fc2=lambda x: x * 2))
        modified, = ffn.H3X2ChunkFeedForward().patch(original, chunks, 4096)
        self.assertEqual(original.patches, {})
        x = torch.arange(rows).reshape(rows, 1)
        with patch.object(ffn.comfy.ops, "linear_input_act", side_effect=lambda linear, value, act: linear(value)):
            out = modified.patches['diffusion_model.blocks.0.mlp.forward'](x)
        self.assertTrue(torch.equal(out, x * 2), "Rows must be preserved exactly once and in order")
        return sizes

    def test_auto_tail_boundaries_and_recorded_shapes(self):
        for rows, expected in [(4097, [2048, 2049]), (5120, [2048, 2048, 1024]),
                               (6143, [2048, 2048, 2047]), (6144, [2048] * 3),
                               (10627, [2048] * 4 + [2435]), (15624, [2048] * 7 + [1288]),
                               (20565, [2048] * 9 + [2133]), (30627, [2048] * 14 + [1955])]:
            with self.subTest(rows=rows):
                self.assertEqual(self.run_partition(rows, 0), expected)

    def test_small_inputs_still_use_one_call(self):
        for rows in (0, 1, 4096):
            with self.subTest(rows=rows):
                self.assertEqual(self.run_partition(rows, 0), [rows])

    def test_existing_equal_count_workflow_is_preserved(self):
        self.assertEqual(self.run_partition(20565, 8), [2571] * 7 + [2568])

    def test_one_chunk_still_disables_the_patch(self):
        original = Model(None)
        result, = ffn.H3X2ChunkFeedForward().patch(original, 1, 4096)
        self.assertIs(result, original)


if __name__ == '__main__':
    unittest.main()
