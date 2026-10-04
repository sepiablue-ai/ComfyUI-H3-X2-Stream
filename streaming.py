"""Adapt ComfyUI's native H3 finalized-chunk output to a bounded NVENC worker.

Spatial/temporal decode and blending stay in ComfyUI. No core monkey patches.
"""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from fractions import Fraction
import json
import math
import os
from pathlib import Path
import tempfile

import av
import torch
import comfy.model_management as mm
import folder_paths


def color_srgb(target):
    # FFmpeg/ISO colour enums: BT.709 primaries/matrix, IEC 61966-2-1 transfer,
    # MPEG limited range. Same settings as ComfyUI's sRGB video writer.
    target.color_primaries = 1
    target.color_trc = 13
    target.colorspace = 1
    target.color_range = 1


def video_tensor(samples):
    value = samples["samples"]
    if getattr(value, "is_nested", False):
        value = value.unbind()[0]
    if not isinstance(value, torch.Tensor) or value.ndim != 5 or value.shape[0] != 1 or value.shape[1] != 24:
        raise ValueError("Expected one H3 video latent [1, 24, T, H, W]. Batch sizes above 1 are unsupported.")
    return value


@contextmanager
def x2_decoder(vae, video, fp16_accumulation):
    vae.throw_exception_if_invalid()
    inner = vae.first_stage_model
    required = ("decoder", "tiling", "tile_size", "tile_overlap_min", "pixel_mean", "pixel_std", "decode_output_shape")
    if not all(hasattr(inner, attr) for attr in required):
        raise ValueError("Connect the MiniMax H3 X2 video VAE, not the conditioning/audio VAE.")
    decoder = inner.decoder
    if tuple(decoder.proj_out.weight.shape) != (12288, 2048):
        raise ValueError("This node supports the X2 Detail v1 projection [12288, 2048].")
    if video.shape[2] < 2 or (video.shape[2] - 2) % 5:
        raise ValueError("Use H3 video lengths on the 5 + 17*k frame grid (for example 124 frames).")
    backend = torch.backends.cuda.matmul
    supports_accumulation = hasattr(backend, "allow_fp16_accumulation")
    if fp16_accumulation and not supports_accumulation:
        raise RuntimeError("This PyTorch lacks allow_fp16_accumulation. Disable the node option or use a supported PyTorch.")
    previous_accumulation = backend.allow_fp16_accumulation if supports_accumulation else None
    saved = {attr: getattr(inner, attr) for attr in ("tiling", "tile_size", "tile_overlap_min", "pixel_mean", "pixel_std")}
    saved_channels = decoder.out_channels
    try:
        with mm.cuda_device_context(vae.device):
            mm.load_models_gpu([vae.patcher], memory_required=vae.memory_used_decode(video.shape, vae.vae_dtype),
                               force_full_load=vae.disable_offload)
            decoder.out_channels = 12
            inner.tiling, inner.tile_size, inner.tile_overlap_min = True, 256, 64
            # Dynamic loading may restore source RGB buffers; expand after loading.
            for attr in ("pixel_mean", "pixel_std"):
                stat = getattr(inner, attr)
                if stat.ndim != 5 or stat.shape[1] not in (3, 12):
                    raise ValueError(f"Unexpected H3 normalization buffer: {attr}, {stat.shape}")
                if stat.shape[1] == 3:
                    setattr(inner, attr, stat.repeat_interleave(4, dim=1))
            if supports_accumulation:
                backend.allow_fp16_accumulation = fp16_accumulation
            yield inner
    finally:
        if supports_accumulation:
            backend.allow_fp16_accumulation = previous_accumulation
        decoder.out_channels = saved_channels
        for attr, value in saved.items():
            setattr(inner, attr, value)


class FrameSink:
    """At most one encoder future plus the current decoded chunk is retained."""
    def __init__(self, output, stream, shape):
        self.output, self.stream, self.shape = output, stream, tuple(shape)
        self.written = 0
        self.pending = None
        self.future = None
        self.worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="h3-x2-nvenc")

    def __getitem__(self, key):
        if len(key) != 5 or any(key[i] != slice(None) for i in (0, 1, 3, 4)):
            raise ValueError("ComfyUI's native H3 output-buffer contract changed.")
        time_slice = key[2]
        if time_slice.step is not None or time_slice.start != self.written:
            raise ValueError("H3 chunks must be written sequentially.")
        self.pending = time_slice.stop - time_slice.start
        return self

    def copy_(self, part):
        mm.throw_exception_if_processing_interrupted()
        batch, channels, count, height, width = part.shape
        if batch != 1 or channels != 12 or count != self.pending or part.dtype != torch.float32:
            raise ValueError("Unexpected H3 finalized chunk shape/dtype.")
        frames = part.reshape(1, 3, 2, 2, count, height, width).permute(0, 4, 5, 2, 6, 3, 1)
        pixels = frames.reshape(count, height * 2, width * 2, 3).mul(255).clamp_(0, 255).to(torch.uint8).cpu().numpy()
        if self.future is not None:
            self.future.result()
        self.future = self.worker.submit(self.encode, pixels)
        self.written += count
        self.pending = None
        return self

    def encode(self, pixels):
        for pixels_frame in pixels:
            frame = av.VideoFrame.from_ndarray(pixels_frame, format="rgb24")
            frame = frame.reformat(format="yuv420p", dst_colorspace=1)
            color_srgb(frame)
            self.output.mux(self.stream.encode(frame))

    def finish(self):
        if self.future is not None:
            self.future.result()
        if self.written != self.shape[2]:
            raise RuntimeError(f"Incomplete decode: {self.written}/{self.shape[2]} frames.")
        self.output.mux(self.stream.encode(None))

    def close(self):
        self.worker.shutdown(wait=True, cancel_futures=True)


def save_video(samples, vae, filename_prefix, fps, nvenc_gpu, cq, fp16_accumulation,
               audio=None, prompt=None, extra_pnginfo=None):
    from comfy.cli_args import args
    if "h264_nvenc" not in av.codecs_available:
        raise RuntimeError("This PyAV/FFmpeg build lacks h264_nvenc. Use an NVENC-enabled build.")
    if vae.device.type != "cuda":
        raise RuntimeError("H3 X2 Stream requires a CUDA VAE device and NVIDIA NVENC.")
    video = video_tensor(samples)
    width, height = int(video.shape[-1]) * 32, int(video.shape[-2]) * 32
    frames = 5 + 17 * ((int(video.shape[2]) - 2) // 5)
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be positive and finite.")
    rate = Fraction(str(fps)).limit_denominator(100000)
    waveform, layout, sample_rate = None, None, None
    if audio is not None:
        sample_rate = int(audio["sample_rate"])
        data = audio["waveform"]
        if sample_rate <= 0 or data.ndim != 3 or data.shape[0] != 1 or data.shape[1] not in (1, 2):
            raise ValueError("Audio must be one mono/stereo waveform [1, channels, samples] at a positive sample rate.")
        waveform = data[0, :, :math.ceil(sample_rate / rate * frames)].float().cpu().contiguous().numpy()
        if not waveform.shape[1]:
            raise ValueError("Audio is empty.")
        layout = "mono" if waveform.shape[0] == 1 else "stereo"
    folder, name, counter, subfolder, _ = folder_paths.get_save_image_path(
        filename_prefix, folder_paths.get_output_directory(), width, height)
    filename = f"{name}_{counter:05}_.mp4"
    target = Path(folder) / filename
    handle, temporary_name = tempfile.mkstemp(prefix=".h3-x2-", suffix=".part.mp4", dir=folder)
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        with av.open(str(temporary), "w", format="mp4") as output:
            if not args.disable_metadata:
                if prompt is not None:
                    output.metadata["prompt"] = json.dumps(prompt)
                if extra_pnginfo:
                    for key, value in extra_pnginfo.items():
                        output.metadata[key] = json.dumps(value)
            stream = output.add_stream("h264_nvenc", rate=rate)
            stream.width, stream.height, stream.pix_fmt, stream.bit_rate = width, height, "yuv420p", 0
            stream.options = {"preset": "p4", "tune": "hq", "rc": "vbr", "cq": str(cq), "gpu": str(nvenc_gpu)}
            color_srgb(stream.codec_context)
            audio_stream = output.add_stream("aac", rate=sample_rate, layout=layout) if waveform is not None else None
            # Fail for an unavailable NVENC device before spending time on decode.
            output.start_encoding()
            with x2_decoder(vae, video, fp16_accumulation) as inner:
                shape = inner.decode_output_shape(video.shape)
                sink = FrameSink(output, stream, shape)
                try:
                    inner.decode(video.to(device=vae.device, dtype=vae.vae_dtype), output_buffer=sink)
                    sink.finish()
                finally:
                    sink.close()
            if audio_stream is not None:
                frame = av.AudioFrame.from_ndarray(waveform, format="fltp", layout=layout)
                frame.sample_rate, frame.pts = sample_rate, 0
                output.mux(audio_stream.encode(frame))
                output.mux(audio_stream.encode(None))
        mm.throw_exception_if_processing_interrupted()
        os.link(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return {"ui": {"images": [{"filename": filename, "subfolder": subfolder, "type": "output"}], "animated": [True]}}
