"""Run with ComfyUI's Python and its root on PYTHONPATH; no generation occurs."""
import importlib.util
from pathlib import Path
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch

SPEC = importlib.util.spec_from_file_location("h3_stream_under_test", Path(__file__).resolve().parents[1] / "streaming.py")
streaming = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(streaming)


class StreamingTests(unittest.TestCase):
    def test_x2_pixel_order_and_rounding_match_pixel_shuffle(self):
        pixels = torch.linspace(0, 1, 12*2*3*4).reshape(1,12,2,3,4)
        captured = []
        sink = streaming.FrameSink(types.SimpleNamespace(mux=lambda packets: None),
                                   types.SimpleNamespace(encode=lambda frame: []), pixels.shape)
        sink.encode = lambda data: captured.append(data.copy())
        try:
            sink[:,:,0:2,:,:].copy_(pixels)
            sink.finish()
        finally:
            sink.close()
        reference = torch.nn.functional.pixel_shuffle(pixels[0].permute(1,0,2,3), 2)
        reference = reference.permute(0,2,3,1).mul(255).clamp(0,255).to(torch.uint8).numpy()
        np.testing.assert_array_equal(captured[0], reference)

    def test_encoder_worker_failure_is_propagated(self):
        sink = streaming.FrameSink(None, None, (1,12,1,2,2))
        def fail(_): raise RuntimeError("encoder failed")
        sink.encode = fail
        try:
            sink[:,:,0:1,:,:].copy_(torch.zeros(1,12,1,2,2))
            with self.assertRaisesRegex(RuntimeError, "encoder failed"):
                sink.finish()
        finally:
            sink.close()

    def test_out_of_order_chunks_rejected(self):
        sink = streaming.FrameSink(None, None, (1,12,5,2,2))
        try:
            with self.assertRaisesRegex(ValueError, "sequentially"):
                sink[:,:,1:2,:,:]
        finally:
            sink.close()

    def test_incomplete_stream_rejected(self):
        sink = streaming.FrameSink(None, None, (1,12,5,2,2))
        try:
            with self.assertRaisesRegex(RuntimeError, "Incomplete decode"):
                sink.finish()
        finally:
            sink.close()

    def test_decoder_state_restored_after_failure(self):
        from contextlib import nullcontext
        decoder = types.SimpleNamespace(proj_out=types.SimpleNamespace(weight=types.SimpleNamespace(shape=(12288,2048))),out_channels=3)
        inner = types.SimpleNamespace(decoder=decoder,tiling=False,tile_size=512,tile_overlap_min=128,
                                      pixel_mean=torch.zeros(1,3,1,1,1),pixel_std=torch.ones(1,3,1,1,1),
                                      decode_output_shape=lambda s:s)
        mean, std = inner.pixel_mean, inner.pixel_std
        vae = types.SimpleNamespace(throw_exception_if_invalid=lambda:None,first_stage_model=inner,device=torch.device('cpu'),
                                    patcher=None,vae_dtype=torch.float16,memory_used_decode=lambda *args:0,disable_offload=False)
        before = torch.backends.cuda.matmul.allow_fp16_accumulation
        with patch.object(streaming.mm,'cuda_device_context',return_value=nullcontext()), patch.object(streaming.mm,'load_models_gpu'):
            with self.assertRaisesRegex(RuntimeError, 'decode failed'):
                with streaming.x2_decoder(vae,torch.zeros(1,24,2,1,1),not before):
                    self.assertEqual(inner.pixel_mean.shape[1],12)
                    raise RuntimeError('decode failed')
        self.assertEqual((inner.tiling,inner.tile_size,inner.tile_overlap_min,decoder.out_channels),(False,512,128,3))
        self.assertIs(inner.pixel_mean,mean)
        self.assertIs(inner.pixel_std,std)
        self.assertEqual(torch.backends.cuda.matmul.allow_fp16_accumulation,before)


if __name__ == '__main__':
    unittest.main()
