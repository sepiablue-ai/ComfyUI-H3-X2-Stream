import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import torch
from safetensors.torch import save_file

SPEC = importlib.util.spec_from_file_location("h3_quant_under_test", Path(__file__).resolve().parents[1] / "quantization.py")
quantization = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(quantization)


class ConversionSafetyTests(unittest.TestCase):
    def test_existing_unrelated_target_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)/'existing.safetensors'
            save_file({'unrelated':torch.ones(2)},str(target))
            before = target.read_bytes()
            with self.assertRaisesRegex(ValueError, 'Existing target'):
                quantization.convert(Path(directory)/'missing.safetensors',target)
            self.assertEqual(target.read_bytes(),before)

    def test_source_cannot_be_destination(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'original.safetensors'
            path.write_bytes(b'preserve original')
            with self.assertRaisesRegex(ValueError,'different files'):
                quantization.convert(path,path)
            self.assertEqual(path.read_bytes(),b'preserve original')

    def test_wrong_projection_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)/'wrong.safetensors'
            save_file({'decoder.proj_out.weight':torch.zeros(2,2,dtype=torch.float16)},str(source))
            with self.assertRaisesRegex(ValueError,'output projection'):
                quantization.inspect_source(source)

    def test_missing_comfy_kitchen_explains_required_comfyui(self):
        with mock.patch.dict(sys.modules, {'comfy_kitchen.tensor.int8': None}):
            with self.assertRaisesRegex(RuntimeError, 'ComfyUI 0.37.0 or newer'):
                quantization.int8_layout()

    def test_package_does_not_install_comfy_kitchen(self):
        root = Path(__file__).resolve().parents[1]
        requirements = [line.split('#')[0].strip() for line in (root/'requirements.txt').read_text(encoding='utf-8').splitlines()]
        self.assertFalse(any(line.startswith('comfy-kitchen') for line in requirements))
        self.assertNotIn('comfy-kitchen>=', (root/'pyproject.toml').read_text(encoding='utf-8'))


if __name__ == '__main__':
    unittest.main()
