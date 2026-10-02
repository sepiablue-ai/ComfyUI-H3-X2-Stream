# Package validation / 公開用パッケージの動作確認

Date / 実施日: **2026-10-02**

## Fresh ComfyUI checkout

A fresh clone of official `Comfy-Org/ComfyUI` was made under the user-designated `C:\comfyui\temp\ComfyUI-H3-X2-Stream`, then pinned to `1568e6cfd04586a4b3c4e1817ea7dde09b1bf9e7` (0.37.0). This package was copied into that clone's `custom_nodes` as an independent directory, not linked back to the experiment runtime. No KJNodes or LatentUpscaler installation was used.

The clone used the existing standard ComfyUI Python environment **read-only**, under the same Windows user and with the same GPU/important flags as the standard launcher. This validates a fresh **ComfyUI checkout and custom-node installation**, not a freshly installed Python dependency environment. Models were read from existing stores through `extra_model_paths.yaml`; generated INT8 weights were written inside the new clone. No global pip installation, standard-runtime edit or benchmark-source edit was performed.

指定された`C:\comfyui\temp`に公式ComfyUIを新規cloneし、上記リビジョンへ固定しました。カスタムノードは独立したコピーを配置し、実験runtimeへのリンクは使っていません。Pythonは標準ComfyUIの既存環境を読み取り利用しています。**ComfyUI本体とカスタムノードの新規導入確認**であり、Python依存関係をゼロからインストールした試験ではありません。モデルは共有先から読み取り、変換結果は新規clone側へ保存しました。

## Actual executions / 実行結果

| Execution / 実行 | Result / 結果 | Comfy execution time |
|---|---|---:|
| Original X2 → INT8 conversion + VAE loading / 初回変換＋読み込み | Success / 成功 | **58.129 s** |
| Case 01, packaged API graph, first full generation after server restart / 再起動後の初回生成 | Success / 成功 | **64.615 s** |
| Case 02, distributed GUI workflow loaded and queued in ComfyUI / 配布GUIワークフローから実行 | Success / 成功 | **59.130 s** |

Each row is **one execution**. Generation rows reused the converted VAE, ran all requested nodes with `--cache-none`, and include H.264/AAC saving. Server startup and conversion are excluded from generation timings. Different cases and warm-up states are shown separately; these two runs are not a repeated statistical comparison. Earlier repeated measurements are in [benchmark.json](benchmark.json) and the READMEs.

各行は**1回の実行**です。生成時は変換済みVAEを再利用し、`--cache-none`で必要なノードを実行しています。H.264/AAC保存を含み、サーバー起動と初回変換は生成時間に含みません。ケースと初期化状態が異なるので、2回を反復比較の平均にはしていません。

## Verification / 検査

- GUI round trip: **every node class, input value and connection** in the Case 02 submitted graph matched the distributed API example after node-ID remapping. This includes the dynamic SLA widget, steps, seed, frame count, crop, prompt and encoder options.
- Both generated files: **1088 × 1920, 124 frames, 24 fps, H.264/yuv420p, BT.709 matrix/primaries, sRGB transfer, AAC stereo at 32 kHz**. Full FFmpeg decoding completed without errors.
- The public converter's **892 tensors exactly matched** the prototype INT8 checkpoint, including non-quantized weights. Only file-level provenance metadata differs.
- **8 automated tests passed:** X2 pixel order/8-bit rounding, encoder-worker error propagation, out-of-order chunks, incomplete streams, restoring VAE/FP16-accumulation state on errors, refusing to overwrite unrelated files, refusing source=destination, and rejecting an incompatible X2 projection.
- No temporary `.h3-x2-*` output files or conversion locks remained after successful runs.

GUIから実行したCase 02は、ID置換を除く全ノード・入力値・接続がAPI版と一致しました。動画2本の全デコード検査、124フレーム・音声・色設定の検査、変換モデル892テンソルの完全一致、異常系を含む8テストを通過しています。画質・音声の主観的な同等性を自動検査だけで確定したものではありません。

## Environment and limits / 環境と制限

Windows; physical GPU 0, NVIDIA GeForce RTX 4070 12 GB; Python 3.13.13; torch 2.14.0+cu130; comfy-kitchen 0.2.35; PyAV 18.1.0; safetensors 0.8.0; ComfyUI frontend 1.53.6. Launch flags included `--disable-pinned-memory`, `--cache-none`, `--disable-api-nodes` and a loopback-only listener.

The existing environment's workflow-template package was 0.11.66 while this ComfyUI asked for 0.11.69. A template-version warning was shown; the distributed graphs loaded and executed successfully. The shared environment was not upgraded for this test.

ComfyUI-Manager installation from a public Git URL was **not tested**, and Registry publication was **not performed**. The package is distributed through [GitHub](https://github.com/sepiablue-ai/ComfyUI-H3-X2-Stream). Python dependency installation on a pristine machine, Linux/other GPUs, other dimensions/fps, and deliberate GPU process-crash recovery remain unverified. The test server was stopped after validation.

既存環境のテンプレート集は0.11.66で、本体要求の0.11.69より古いという警告が出ましたが、配布グラフの読み込み・実行は成功しています。共有環境の更新は行っていません。公開Git URLからのManager導入、Registry登録、クリーンなPython依存環境、Linux／別GPU、別解像度・fps、GPUプロセス強制終了からの復旧は未検証です。確認用サーバーは作業後に停止しました。
