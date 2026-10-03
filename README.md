# ComfyUI H3 X2 Stream

[日本語](README_ja.md) | English

Prepare an INT8 version of the MiniMax H3 X2 Detail VAE inside ComfyUI, then decode and save video with a bounded asynchronous NVIDIA NVENC encoder. Includes an H3 FFN chunking node and ready-to-load 4-step examples.

On an **RTX 4070 12 GB**, the development configuration generated **1088 × 1920, 124-frame, 24 fps video with audio in about 60 seconds**. This is a measured configuration, not a speed guarantee for arbitrary models, machines, or resolutions. See the results below and [Sayaka Benchmark](https://sayakabenchmark.pages.dev/) for the underlying workflow collection and comparisons.

## What is included

| Node | Purpose |
|---|---|
| **H3 X2 Prepare INT8 VAE** | Converts the original X2 Detail v1 checkpoint once, saves it locally, and returns a usable VAE. Subsequent executions reuse the converted file. |
| **H3 X2 Decode + Stream Save** | Preserves ComfyUI's native H3 spatial/temporal blending, unpacks X2 RGB, and overlaps the next decode chunk with H.264 NVENC encoding. Optional audio is saved as AAC. |
| **H3 Chunk FeedForward (X2 Stream)** | Splits H3 FFN token work into 2048-token chunks, merging a small final chunk. Default: `chunks=0`, threshold 4096. |

No additional KJNodes or LatentUpscaler installation is required for the included examples. No ComfyUI core files are overwritten. The preparation node does not download models or install Python packages.

## Requirements

- A recent ComfyUI with native MiniMax H3, `BlockSparseAttention`, ConvRot INT8 support, and the H3 `decode(..., output_buffer=...)` interface. **Tested: ComfyUI 0.37.0, commit `1568e6cfd04586a4b3c4e1817ea7dde09b1bf9e7`.** This exact revision is the reproducible reference; compatibility with every later release is not implied.
- NVIDIA CUDA GPU and a working `h264_nvenc` encoder in PyAV/FFmpeg. Tested on Windows and RTX 4070 12 GB. AMD, Apple, CPU-only execution and Linux have not been validated.
- Tested libraries: **Python 3.13.13, PyTorch 2.14.0+cu130, comfy-kitchen 0.2.35, PyAV 18.1.0, safetensors 0.8.0**. Use the normal ComfyUI installation instructions for CUDA/PyTorch. This package does not automatically replace PyTorch.
- `fp16_accumulation=true` requires PyTorch's `torch.backends.cuda.matmul.allow_fp16_accumulation`. If your build does not expose it, disable the node option. The published timings used it enabled.
- About **5.25 GB** for the original X2 VAE plus **2.83 GB** for the converted VAE, in addition to the other H3 models. Conversion also temporarily holds the checkpoint on the CPU. The tested PC had about 48 GiB RAM; a lower RAM minimum has not been established.

## Install

Run from the ComfyUI directory:

```shell
git clone https://github.com/sepiablue-ai/ComfyUI-H3-X2-Stream.git custom_nodes/ComfyUI-H3-X2-Stream
```

Alternatively, extract this [repository's ZIP](https://github.com/sepiablue-ai/ComfyUI-H3-X2-Stream/archive/refs/heads/main.zip) to `ComfyUI/custom_nodes/ComfyUI-H3-X2-Stream/`. Install `requirements.txt` **with the Python belonging to that ComfyUI**, then restart ComfyUI. For example:

```powershell
# Portable Windows installation: run from ComfyUI_windows_portable
.\python_embeded\python.exe -m pip install -r .\ComfyUI\custom_nodes\ComfyUI-H3-X2-Stream\requirements.txt

# Venv installation: run from ComfyUI, with its own venv activated
python -m pip install -r custom_nodes/ComfyUI-H3-X2-Stream/requirements.txt
```

Use the command appropriate to your installation, not both. Do not install into a system/global Python.

**ComfyUI-Manager:** use `https://github.com/sepiablue-ai/ComfyUI-H3-X2-Stream` for Git-URL installation where your Manager's security configuration permits it. This route can install the code and requirements, but was not part of the validation run; the manual installation above was tested. Registry publication is optional and this package is not currently registered. Manager does not perform GPU weight conversion itself: the **Prepare INT8 VAE node does that when the workflow is queued**. There is no conversion in `install.py` and no need to run a terminal conversion command. See [Manager configuration](https://docs.comfy.org/manager/configuration) and [Registry publishing](https://docs.comfy.org/registry/publishing).

## Models

Read the respective model terms before downloading. No weights are bundled or automatically downloaded.

| File / source | Put in | Role |
|---|---|---|
| [MiniMax-H3-X2-Detail-v1.safetensors](https://huggingface.co/speach1sdef178/MiniMax-H3-X2-Detail-VAE) | `models/vae/` | Original X2 VAE, input to local conversion |
| [minimax_h3_fused_refdelta_r1024_turbo8_mystic07_int8_convrot.safetensors](https://huggingface.co/MATLOWAI/minimax-h3-fused-turbo-int8-convrot/tree/main/diffusion_models) | `models/diffusion_models/` | Exact MATLOW fused diffusion model used in the measurements |
| [qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/tree/main/text_encoders) | `models/text_encoders/` | Text encoder |
| [minimax_h3_video_vae_int8_convrot.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/tree/main/vae) | `models/vae/` | Original H3 video VAE for **image conditioning** |
| [minimax_h3_audio_vae_fp32.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/tree/main/vae) | `models/vae/` | Audio decoding |

The X2 VAE belongs on the final video decode branch. Keep the ordinary video VAE on the `MiniMaxH3ImageToVideo` conditioning branch. The examples already separate these paths. The measured model is a particular fused model; swapping to another H3 model can require different steps, attention settings and guidance.

## Generate in ComfyUI

1. Copy the two images from [`examples/assets`](examples/assets) into `ComfyUI/input/` without changing their names.
2. Drag [`examples/fl2va_4step_case01.json`](examples/fl2va_4step_case01.json) or [`case02`](examples/fl2va_4step_case02.json) onto ComfyUI.
3. Select the installed models. In **Prepare INT8 VAE**, select the original `MiniMax-H3-X2-Detail-v1.safetensors`.
4. Set the stream-save node's **NVENC GPU index**. The default is 0. This value is passed to FFmpeg/NVENC; check the encoder's device enumeration on multi-GPU systems. Only GPU 0 was validated here.
5. Queue the workflow. The first run verifies the original model's SHA-256 and performs conversion. Later runs reuse the file. For preparation without video generation, use [`00_prepare_int8.json`](examples/00_prepare_int8.json).

Converted files are written only to the active ComfyUI's `models/vae/h3_x2_stream/`, even when the source is in an `extra_model_paths.yaml` directory. The original file is never modified. Existing output models are reused only if their conversion metadata matches this recipe; an unrelated file at the same name causes an error.

Videos are saved to the configured ComfyUI output directory, normally `output/H3_X2_Stream/`. Failed encodes remove their temporary video instead of leaving a completed-looking MP4. Only one video latent and mono/stereo audio are supported per execution. No `IMAGE` output is produced because streaming avoids allocating the whole decoded clip; use a conventional decoder when downstream image-processing nodes are needed.

The adapter derives spatial dimensions and frame count from the latent, and exposes fps and encoder quality. The **validated preset** is 544 × 960 sampling → 1088 × 1920 output, 124 frames at 24 fps. Additional generation sizes of 512 × 512, 768 × 512 and 768 × 1024 were tested with X2 output at the same frame count and fps. Other frame rates have not been validated. Native 256-pixel tiles with 64-pixel overlap are retained. Video lengths must follow H3's 5 + 17*k frame grid.

The FFN node's `chunks=0` selects automatic 2048-token chunks. If the last chunk has fewer than 1024 tokens, it is merged into the preceding chunk; for example, 2048 + 85 becomes 2133. Inputs at or below `seq_threshold` (default 4096) run without splitting. `chunks=1` disables the patch; values of 2 or more retain the previous equal-count splitting. Existing saved workflows keep their value: set `chunks` to **0** to enable the new mode. Bundled examples already use 0.

API examples are under [`examples/api`](examples/api). Advanced users may run `quantization.py SOURCE OUTPUT --device cuda:0` using ComfyUI's Python; the GUI preparation node is the normal path.

## RTX 4070 measurements

### 2048-token FFN update — 2026-10-03

On the same RTX 4070 12 GB, a 4-step FL2VA Case 01 comparison at 544 × 960 reduced median execution time from **60.319 s** (8 chunks) to **59.212 s** (automatic 2048-token chunks), with three runs per mode. Median sampler time fell from 38.700 s to 37.509 s. The automatic-mode runs took 59.212 / 59.284 / 58.884 s. Model, prompt, seed, steps and attention settings were held fixed within each comparison.

Three additional first-frame-conditioned cases used different images and prompts, with 124 frames at 24 fps and generated audio:

| Generation → X2 output | 8 chunks, seconds | Automatic 2048, seconds |
|---|---:|---:|
| 512 × 512 → 1024 × 1024 | 50.491 | 48.951 |
| 768 × 512 → 1536 × 1024 | 49.856 | 49.152 |
| 768 × 1024 → 1536 × 2048 | 83.444 | 82.295 |

These additional sizes have **one run per mode**, so the differences are indicative, not repeated-run estimates. Timing covers `execution_start` through `execution_success`, including decoding and saving, but excludes server startup and media checks. There were no execution-cache hits; OS/compiler cache and initialization effects were not isolated. All six videos passed full decoding without OOM. The square pair had identical decoded RGB and audio PCM; the other pairs differed, so bit-identical output is not guaranteed. SLA used 5% keep with `min_tokens=12288`; the square case was below that threshold and used dense attention in both modes.

### Original streaming measurements — 2026-10-02

Measured on **2026-10-02**, Windows, RTX 4070 12 GB. Seed 43, 124 frames / 24 fps (about 5.17 s), native audio, 544 × 960 sampling and 1088 × 1920 X2 output. The baseline is FL2VA `017-matlow-fused4-sla5-x2vae` from [Sayaka Benchmark](https://sayakabenchmark.pages.dev/).

| Configuration | Case 01, seconds | Case 02, seconds | Under 60 s |
|---|---:|---:|---:|
| Baseline: FP16 X2, FFN chunks 4 | 86.725 (1 run) | 82.680 (1 run) | 0/2 |
| **4 steps, INT8 X2, FFN chunks 8, async NVENC** | 59.006 / 59.794 / 60.300; **mean 59.700** | 59.634 / 59.783 / 59.962; **mean 59.793** | **5/6** |
| 3-step experiment | 50.918 / 51.864 / 52.007; mean 51.596 | 51.387 / 51.943 / 51.737; mean 51.689 | 6/6 |

These are **development/prototype measurements**, preserved in [`docs/benchmark.json`](docs/benchmark.json). Packaging changes are tested separately in [`docs/validation.md`](docs/validation.md). Timings run from ComfyUI `execution_start` to `execution_success` after server startup, with graph caching disabled (`--cache-none`) and pinned memory disabled (`--disable-pinned-memory`). Server startup, one-time conversion and post-run media verification are excluded. They are not cold application-launch or download timings.

The tested sampler was `res_multistep` / `simple`, BasicGuider, video/audio shifts 12/3, SLA 5%, `min_tokens=12288`, `extra_tokens=256`, and exact conditioning/audio rows. The VAE option enabled FP16 accumulation only during decode and restored it afterward. NVENC used H.264, p4/hq, VBR CQ18, yuv420p with sRGB transfer/BT.709 matrix and primaries; audio used AAC.

**Four steps is the shipped preset.** The three-step experiment changed dark sleeves into light sleeves in Case 02, so it is not the recommended default and has no bundled production workflow.

INT8 conversion quantizes only 144 decoder block Linear weights using per-channel INT8 ConvRot, group size 256. The encoder, norms, biases and final FP16 X2 projection are preserved. Development comparisons on seven identical-latent raw frames per case yielded mean PSNR 57.75 / 57.83 dB versus the FP16 reference. Async and synchronous streaming produced identical decoded RGB for all 124 frames and identical decoded audio in the tested comparison. These checks do not establish perceptual equality for every prompt; final visual/audio acceptance remains a human judgment.

## Troubleshooting and removal

- **Missing H3 / sparse-attention node:** use the tested ComfyUI revision or an appropriate newer version. The example's sparse node is part of ComfyUI core.
- **Missing CUDA/INT8 kernel:** check your ComfyUI Python, CUDA-enabled PyTorch and Comfy Kitchen combination. The INT8 path is not a generic CPU fallback.
- **NVENC encoder/device error:** confirm the selected physical GPU and driver support H.264 NVENC and that PyAV includes the encoder. There is no silent software-encoding fallback.
- **Source hash mismatch:** this converter intentionally accepts only the original X2 Detail v1 release, SHA-256 `2296840f4acedcaa976688e7d7b97f7bf570b136e400385d3f46224011897aac`.
- **Existing target is incompatible:** keep the existing file and choose a different output filename. Files from the earlier private experiment do not carry this public converter's metadata.
- **Conversion interrupted by a process crash:** a `.lock` or `.h3-x2-*` temporary file can remain in `models/vae/h3_x2_stream`. After confirming that no conversion process is running, remove only the stale file and retry. Normal errors/cancellation clean up automatically.
- **Out of memory:** stop other GPU workloads or choose a smaller generation size. This node deliberately avoids ComfyUI's generic tiled OOM fallback, which would change the H3 decode path.
- To uninstall, stop ComfyUI and remove this custom-node directory. Converted models are separate files under `models/vae/h3_x2_stream`; keep them or delete them explicitly. No global settings need reverting.

## Attribution

GPL-3.0 code. See [LICENSE](LICENSE) and [NOTICE.md](NOTICE.md) for KJNodes attribution, ComfyUI/Comfy Kitchen dependencies, X2 model authorship and separate model terms. The original model's license also applies to local converted weights; this code license does not grant new rights over those weights.
