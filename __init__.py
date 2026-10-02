"""ComfyUI-H3-X2-Stream: local INT8 preparation and streaming H3 X2 output."""

from .nodes import H3X2PrepareINT8VAE, H3X2StreamSave
from .ffn import H3X2ChunkFeedForward

NODE_CLASS_MAPPINGS = {
    "H3X2PrepareINT8VAE": H3X2PrepareINT8VAE,
    "H3X2StreamSave": H3X2StreamSave,
    "H3X2ChunkFeedForward": H3X2ChunkFeedForward,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "H3X2PrepareINT8VAE": "H3 X2 Prepare INT8 VAE",
    "H3X2StreamSave": "H3 X2 Decode + Stream Save",
    "H3X2ChunkFeedForward": "H3 Chunk FeedForward (X2 Stream)",
}
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
