"""One-time, local conversion; no downloads, pip calls, or global configuration."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

import torch
from safetensors import safe_open
from safetensors.torch import save_file

RECIPE = "h3-x2-stream.decoder-int8-convrot.v1"
SOURCE_SHA256 = "2296840f4acedcaa976688e7d7b97f7bf570b136e400385d3f46224011897aac"
DEFAULT_OUTPUT = "MiniMax-H3-X2-Detail-v1-decoder-int8-convrot.safetensors"


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def inspect_source(path):
    with safe_open(str(path), framework="pt", device="cpu") as model:
        if any(k.endswith(".comfy_quant") for k in model.keys()):
            raise ValueError("Select the original FP16 X2 Detail v1 VAE, not an already quantized model.")
        head = model.get_tensor("decoder.proj_out.weight")
        if tuple(head.shape) != (12288, 2048) or head.dtype != torch.float16:
            raise ValueError("Expected X2 Detail v1 FP16 output projection [12288, 2048].")
        linears = [k for k in model.keys() if k.startswith("decoder.transformer_blocks.")
                   and k.endswith(".weight") and len(model.get_slice(k).get_shape()) == 2]
        if len(linears) != 144:
            raise ValueError(f"Expected 144 decoder Linear weights; found {len(linears)}.")
        if any(model.get_slice(k).get_shape()[1] % 256 for k in linears):
            raise ValueError("Decoder dimensions are incompatible with ConvRot group size 256.")
        return linears


def inspect_converted(path):
    with safe_open(str(path), framework="pt", device="cpu") as model:
        meta = model.metadata() or {}
        if meta.get("h3_x2_stream_recipe") != RECIPE or meta.get("source_sha256") != SOURCE_SHA256:
            raise ValueError("Existing target was not made by this converter from the supported original VAE. Choose a new output name.")
        keys = model.keys()
        if len([k for k in keys if k.endswith(".comfy_quant")]) != 144:
            raise ValueError("Converted VAE has incomplete quantization metadata.")
        head = model.get_tensor("decoder.proj_out.weight")
        if tuple(head.shape) != (12288, 2048) or head.dtype != torch.float16:
            raise ValueError("Converted VAE does not preserve the FP16 X2 output head.")
        return meta


def convert(source, output, *, device="cuda:0", progress=None, check_cancel=None):
    """Validate the official source, then atomically publish a new file. Never overwrite."""
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output:
        raise ValueError("Input and output must be different files.")
    if output.exists():
        meta = inspect_converted(output)
        return {"status": "reused", "output": str(output), "bytes": output.stat().st_size, "metadata": meta}
    if not source.is_file():
        raise FileNotFoundError(source)
    if not torch.cuda.is_available() or torch.device(device).type != "cuda":
        raise RuntimeError("INT8 ConvRot conversion requires a CUDA GPU and a CUDA-enabled ComfyUI environment.")
    from comfy_kitchen.tensor.int8 import TensorWiseINT8Layout

    linears = set(inspect_source(source))
    if sha256(source) != SOURCE_SHA256:
        raise ValueError("Source SHA-256 does not match MiniMax-H3-X2-Detail-v1. Other checkpoints are not supported.")
    if check_cancel:
        check_cancel()
    output.parent.mkdir(parents=True, exist_ok=True)
    lock = output.with_suffix(output.suffix + ".lock")
    temporary = None
    begin = time.perf_counter()
    # Exclusive lock protects concurrent conversions; only our own lock is removed.
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(f"pid={os.getpid()}\n")
    try:
        if output.exists():
            raise FileExistsError(output)
        state = {}
        completed = 0
        with torch.inference_mode(), safe_open(str(source), framework="pt", device="cpu") as model:
            metadata = dict(model.metadata() or {})
            for key in model.keys():
                if check_cancel:
                    check_cancel()
                weight = model.get_tensor(key)
                if key in linears:
                    quantized, params = TensorWiseINT8Layout.quantize(
                        weight.to(device), per_channel=True, convrot=True, convrot_groupsize=256)
                    state[key] = quantized.cpu().contiguous()
                    state[key[:-6] + "weight_scale"] = params.scale.cpu().contiguous()
                    marker = {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256}
                    state[key[:-6] + "comfy_quant"] = torch.tensor(list(json.dumps(marker).encode()), dtype=torch.uint8)
                    del quantized, params
                    completed += 1
                    if progress:
                        progress(completed, 144)
                else:
                    state[key] = weight
            metadata.update(h3_x2_stream_recipe=RECIPE, source_sha256=SOURCE_SHA256,
                            quantized_linears="144", preserved="encoder; norms; biases; FP16 X2 output projection")
            handle, name = tempfile.mkstemp(prefix=".h3-x2-", suffix=".safetensors", dir=output.parent)
            os.close(handle)
            temporary = Path(name)
            save_file(state, str(temporary), metadata=metadata)
        inspect_converted(temporary)
        if check_cancel:
            check_cancel()
        # Hard-link publication is atomic and fails if a target appeared meanwhile.
        os.link(temporary, output)
        return {"status": "converted", "output": str(output), "bytes": output.stat().st_size,
                "seconds": round(time.perf_counter() - begin, 3), "source_sha256": SOURCE_SHA256,
                "output_sha256": sha256(output), "quantized_linears": completed}
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        lock.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Original MiniMax-H3-X2-Detail-v1.safetensors")
    parser.add_argument("output", type=Path, help="New .safetensors file (never overwritten)")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    print(json.dumps(convert(args.source, args.output, device=args.device), indent=2))


if __name__ == "__main__":
    main()
