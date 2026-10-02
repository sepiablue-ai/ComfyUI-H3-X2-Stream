import json
from pathlib import Path

import comfy.model_management as mm
import comfy.sd
import comfy.utils
import folder_paths

from .quantization import DEFAULT_OUTPUT, convert
from .streaming import save_video


class H3X2PrepareINT8VAE:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "source_vae": (folder_paths.get_filename_list("vae"),),
            "output_name": ("STRING", {"default": DEFAULT_OUTPUT}),
        }}

    RETURN_TYPES = ("VAE", "STRING")
    RETURN_NAMES = ("vae", "conversion_report")
    FUNCTION = "prepare"
    CATEGORY = "H3 X2 Stream"
    OUTPUT_NODE = True
    DESCRIPTION = ("Prepare the official X2 Detail v1 VAE once, then reuse it. "
                   "Writes to models/vae/h3_x2_stream. No download or package install. "
                   "Connect this VAE only to the X2 decode node; keep the conditioning VAE separate.")

    def prepare(self, source_vae, output_name):
        if (not output_name or Path(output_name).name != output_name or "/" in output_name or "\\" in output_name
                or ":" in output_name or not output_name.endswith(".safetensors")):
            raise ValueError("output_name must be a filename ending in .safetensors, without directories.")
        source = folder_paths.get_full_path_or_raise("vae", source_vae)
        # Write only to this ComfyUI's model root, never to a shared extra-model path.
        destination = Path(folder_paths.models_dir) / "vae" / "h3_x2_stream" / output_name
        progress = comfy.utils.ProgressBar(144)
        if not destination.exists():
            mm.unload_all_models()
            mm.soft_empty_cache()
        report = convert(source, destination, device=str(mm.get_torch_device()),
                         progress=progress.update_absolute, check_cancel=mm.throw_exception_if_processing_interrupted)
        state, metadata = comfy.utils.load_torch_file(str(destination), safe_load=True, return_metadata=True)
        vae = comfy.sd.VAE(sd=state, metadata=metadata)
        vae.throw_exception_if_invalid()
        text = json.dumps(report, ensure_ascii=False, indent=2)
        return {"ui": {"text": [text]}, "result": (vae, text)}


class H3X2StreamSave:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "samples": ("LATENT",), "vae": ("VAE",),
            "filename_prefix": ("STRING", {"default": "H3_X2_Stream/video"}),
            "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0, "step": 0.01}),
            "nvenc_gpu": ("INT", {"default": 0, "min": 0, "max": 31,
                                   "tooltip": "Device index passed to FFmpeg/NVENC. Default 0; verify encoder enumeration on multi-GPU systems."}),
            "cq": ("INT", {"default": 18, "min": 0, "max": 51}),
            "fp16_accumulation": ("BOOLEAN", {"default": True}),
        }, "optional": {"audio": ("AUDIO",)}, "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"}}

    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "H3 X2 Stream"
    DESCRIPTION = ("Native H3 X2 decode with 256/64 spatial tiles, async H.264 NVENC and optional AAC. "
                   "One video per execution. Tested: RTX 4070, 1088x1920, 124 frames, 24 fps.")

    def save(self, **kwargs):
        return save_video(**kwargs)
