"""Token-chunked H3 FFN, adapted from kijai/ComfyUI-KJNodes (GPL-3.0).

Upstream: nodes/minimax_nodes.py, MiniMaxChunkFeedForward.
See NOTICE.md for provenance. Node IDs are deliberately distinct.
"""
from types import MethodType

import torch
import comfy.ops


class H3X2ChunkFeedForward:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "chunks": ("INT", {"default": 0, "min": 0, "max": 64, "tooltip": "0: 2048-token chunks, merging a tail below 1024 tokens. 1: disable. 2+: equal-count chunks."}),
            "seq_threshold": ("INT", {"default": 4096, "min": 256, "max": 262144, "step": 256}),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "H3 X2 Stream"

    def patch(self, model, chunks, seq_threshold):
        if chunks == 1:
            return (model,)
        patched = model.clone()
        blocks = getattr(patched.get_model_object("diffusion_model"), "blocks", None)
        if not blocks or any(not hasattr(b, "mlp") or not hasattr(b.mlp, "fc2") for b in blocks):
            raise ValueError("H3 Chunk FeedForward requires a MiniMax H3 diffusion model.")

        def forward(module, x):
            if x.shape[0] <= seq_threshold:
                return comfy.ops.linear_input_act(module.fc2, module.fc1(x), "swiglu")
            result = torch.empty_like(x)
            offset = 0
            parts = torch.split(x, 2048, dim=0) if chunks == 0 else torch.chunk(x, chunks, dim=0)
            if chunks == 0 and len(parts) > 1 and len(parts[-1]) < 1024:
                parts = (*parts[:-2], x[-(len(parts[-2]) + len(parts[-1])):])
            for part in parts:
                result[offset:offset + len(part)] = comfy.ops.linear_input_act(module.fc2, module.fc1(part), "swiglu")
                offset += len(part)
            return result

        for index, block in enumerate(blocks):
            patched.add_object_patch(f"diffusion_model.blocks.{index}.mlp.forward", MethodType(forward, block.mlp))
        return (patched,)
