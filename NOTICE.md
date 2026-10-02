# Attribution and model terms

Code in this repository is distributed under GPL-3.0; see LICENSE.

- `ffn.py` adapts MiniMaxChunkFeedForward from [kijai/ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes), `nodes/minimax_nodes.py` (GPL-3.0). The inspected source SHA-256 is `acbfdd2c25ebec34b1ade23d4856931209a9e1d5b690b810f2cef0af47832642`. Changes: distinct node ID, isolated closure settings, explicit incompatible-model error, smaller import surface.
- [ComfyUI](https://github.com/Comfy-Org/ComfyUI) supplies the native H3 decoder, model management, output paths, and inference operations. This package calls the native decoder; it does not redistribute or replace ComfyUI's temporal/spatial blending code.
- X2 packed RGB decoding was informed by the public model description and [TripleHeadedMonkey/ComfyUI-MiniMaxH3_LatentUpscaler](https://github.com/TripleHeadedMonkey/ComfyUI-MiniMaxH3_LatentUpscaler). Its source files are not bundled here. The streaming adapter is implemented against ComfyUI's native output-buffer interface.
- INT8 ConvRot conversion uses [Comfy Kitchen](https://github.com/Comfy-Org/comfy-kitchen).
- The original X2 VAE is by [speach1sdef178](https://huggingface.co/speach1sdef178/MiniMax-H3-X2-Detail-VAE). Model weights are not included. The original and locally converted weights remain subject to the model's [LICENSE](https://huggingface.co/speach1sdef178/MiniMax-H3-X2-Detail-VAE/blob/main/LICENSE) and [NOTICE](https://huggingface.co/speach1sdef178/MiniMax-H3-X2-Detail-VAE/blob/main/NOTICE), including applicable MiniMax H3 terms. The code license does not replace those terms.
- Benchmark prompts/settings originate from the author's [Sayaka Benchmark](https://sayakabenchmark.pages.dev/). No benchmark source directory is required at runtime.
