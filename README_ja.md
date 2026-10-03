# ComfyUI H3 X2 Stream

日本語 | [English](README.md)

MiniMax H3のX2 Detail VAEをComfyUI内でINT8へ変換し、デコードとNVIDIA NVENCによる動画保存を並行実行するカスタムノードです。FFN分割ノードと、そのまま読み込める4ステップ版ワークフローも付属します。

**RTX 4070 12 GBで、1088 × 1920・124フレーム・24 fps・音声付き動画を約60秒で生成**した構成を元にしています。任意のモデル・PC・解像度で1分未満になる保証ではありません。元のワークフローや比較動画は、[さやかベンチマーク](https://sayakabenchmark.pages.dev/)を参照してください。

## 入っているもの

| ノード | 役割 |
|---|---|
| **H3 X2 Prepare INT8 VAE** | 元のX2 Detail v1 VAEを初回だけ変換・保存し、そのまま使えるVAEを出力します。2回目以降は保存済みファイルを再利用します。 |
| **H3 X2 Decode + Stream Save** | ComfyUI標準H3の空間・時間方向のブレンドを保ち、X2 RGB展開とH.264 NVENC保存を行います。次のチャンクのデコード中に前のチャンクをエンコードし、音声は任意でAAC保存します。 |
| **H3 Chunk FeedForward (X2 Stream)** | H3のFFNを2048トークンずつに分割し、小さい末尾を直前へ結合します。初期値は`chunks=0`・閾値4096です。 |

付属例では、KJNodesやLatentUpscaler一式の追加インストールは不要です。ComfyUI本体の書き換えも行いません。変換ノードはモデルのダウンロードやPythonパッケージのインストールを行いません。

## 動作条件

- MiniMax H3、`BlockSparseAttention`、ConvRot INT8、H3の`decode(..., output_buffer=...)`に対応するComfyUI。**確認済みはComfyUI 0.37.0、commit `1568e6cfd04586a4b3c4e1817ea7dde09b1bf9e7`**です。再現時の基準はこの版です。以後の全バージョンでの互換性を保証するものではありません。
- NVIDIA CUDA GPUと、PyAV/FFmpegの`h264_nvenc`。Windows・RTX 4070 12 GBで確認しました。AMD、Apple、CPUのみの実行、Linuxは未検証です。
- 確認済みライブラリ：**Python 3.13.13、PyTorch 2.14.0+cu130、comfy-kitchen 0.2.35、PyAV 18.1.0、safetensors 0.8.0**。CUDA/PyTorchは通常のComfyUI導入手順で準備してください。このパッケージがPyTorchを自動交換することはありません。
- `fp16_accumulation=true`にはPyTorchの`torch.backends.cuda.matmul.allow_fp16_accumulation`が必要です。対応しない版ではノードの設定をfalseにしてください。掲載した実測はtrueです。
- 元のX2 VAE約**5.25 GB**と変換後約**2.83 GB**、そのほかのH3モデルの保存領域が必要です。変換中はCPUメモリにもチェックポイントを保持します。検証PCのRAMは約48 GiBで、それより小さいRAMでの下限は未確認です。

## インストール

ComfyUIのフォルダから、次のコマンドでcloneします。

```shell
git clone https://github.com/sepiablue-ai/ComfyUI-H3-X2-Stream.git custom_nodes/ComfyUI-H3-X2-Stream
```

または、[このrepoのZIP](https://github.com/sepiablue-ai/ComfyUI-H3-X2-Stream/archive/refs/heads/main.zip)を`ComfyUI/custom_nodes/ComfyUI-H3-X2-Stream/`へ展開します。**使用するComfyUI専用のPython**で`requirements.txt`をインストールし、ComfyUIを再起動してください。例：

```powershell
# Windows portable版：ComfyUI_windows_portableから実行
.\python_embeded\python.exe -m pip install -r .\ComfyUI\custom_nodes\ComfyUI-H3-X2-Stream\requirements.txt

# venv版：ComfyUIから、その環境のvenvを有効にして実行
python -m pip install -r custom_nodes/ComfyUI-H3-X2-Stream/requirements.txt
```

自分の環境に合う方だけを使ってください。システムPythonやグローバル環境には入れません。

**ComfyUI-Managerについて：** Git URLからの導入がセキュリティ設定で許可されている環境では、`https://github.com/sepiablue-ai/ComfyUI-H3-X2-Stream`を指定できます。コードと依存関係を導入できる経路ですが、今回この経路の実機確認は行っていません。検証したのは上記の手動配置です。Registry登録は任意で、このパッケージは現在未登録です。Manager自体がGPUで重みを変換するわけではなく、**ワークフローをキューに入れるとPrepare INT8 VAEノードが変換します**。インストール時の`install.py`では変換しません。通常の利用ではターミナルで変換コマンドを実行する必要はありません。[Manager設定](https://docs.comfy.org/manager/configuration)／[Registry公開手順](https://docs.comfy.org/registry/publishing)。

## 用意するモデル

入手先の利用条件を確認してください。モデル重みは同梱せず、自動ダウンロードもしません。

| ファイル・入手先 | 配置先 | 用途 |
|---|---|---|
| [MiniMax-H3-X2-Detail-v1.safetensors](https://huggingface.co/speach1sdef178/MiniMax-H3-X2-Detail-VAE) | `models/vae/` | INT8変換元となるX2 VAE |
| [minimax_h3_fused_refdelta_r1024_turbo8_mystic07_int8_convrot.safetensors](https://huggingface.co/MATLOWAI/minimax-h3-fused-turbo-int8-convrot/tree/main/diffusion_models) | `models/diffusion_models/` | 実測に使ったMATLOWの融合モデル |
| [qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/tree/main/text_encoders) | `models/text_encoders/` | テキストエンコーダー |
| [minimax_h3_video_vae_int8_convrot.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/tree/main/vae) | `models/vae/` | **入力画像の条件付け**に使う通常のH3動画VAE |
| [minimax_h3_audio_vae_fp32.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/tree/main/vae) | `models/vae/` | 音声デコード |

X2 VAEは最後の動画デコードに使います。`MiniMaxH3ImageToVideo`への入力画像を条件付けするVAEは通常版のままにしてください。付属ワークフローでは接続を分けてあります。実測は特定の融合モデルで行ったもので、別のH3モデルではステップ数・Attention・Guidanceなどの調整が必要になる場合があります。

## ComfyUIだけで変換・生成する

1. [`examples/assets`](examples/assets)の画像2枚を、名前を変えず`ComfyUI/input/`へコピーします。
2. [`examples/fl2va_4step_case01.json`](examples/fl2va_4step_case01.json)または[`case02`](examples/fl2va_4step_case02.json)をComfyUIへドラッグします。
3. 各モデルを選びます。**Prepare INT8 VAE**では元の`MiniMax-H3-X2-Detail-v1.safetensors`を選びます。
4. 保存ノードの**NVENC GPU番号**を指定します。初期値は0で、FFmpeg/NVENCへ渡す番号です。複数GPU環境ではエンコーダー側のデバイス番号を確認してください。今回はGPU 0のみ検証しています。
5. 実行します。初回は元モデルのSHA-256確認とINT8変換が入り、2回目以降は変換済みファイルを再利用します。動画を生成せず準備だけ行う場合は[`00_prepare_int8.json`](examples/00_prepare_int8.json)を使います。

変換先は起動中のComfyUIの`models/vae/h3_x2_stream/`です。元モデルが`extra_model_paths.yaml`で共有されていても、共有元には書き込みません。元ファイルは変更せず、既存の変換済みファイルも上書きしません。同名のファイルが今回の変換形式と一致しない場合はエラーになります。

動画の保存先はComfyUIで設定したoutput配下で、標準では`output/H3_X2_Stream/`です。保存に失敗した場合は一時動画を取り除き、完成したように見えるMP4を残しません。1回の実行は動画1本、音声はモノラルまたはステレオに対応します。動画全体の画像テンソルを保持しないため`IMAGE`出力はありません。後段に画像処理ノードを接続したい場合は通常のデコーダーを使ってください。

解像度とフレーム数はlatentから取得し、fpsとエンコーダー品質はノードで指定できます。**検証済みの設定**は544 × 960生成 → 1088 × 1920出力、124フレーム、24 fpsです。追加で512 × 512、768 × 512、768 × 1024生成も、同じフレーム数・fpsとX2出力で検証しました。他のfpsは未検証です。空間タイルは標準の256／重なり64を保ちます。動画長はH3の5 + 17*kフレームの規則に従う必要があります。

FFNノードの`chunks=0`は2048トークン単位の自動分割です。末尾が1024トークン未満なら直前の分割に結合します。例えば2048＋85は2133になります。入力が`seq_threshold`（初期値4096）以下なら分割しません。`chunks=1`はパッチ無効、2以上は従来どおり指定数での分割です。保存済みワークフローの値は維持されるため、新方式を使う場合は`chunks`を**0**に変更してください。同梱例は0へ更新済みです。

API用JSONは[`examples/api`](examples/api)にあります。上級者向けに、ComfyUI専用Pythonから`quantization.py SOURCE OUTPUT --device cuda:0`でも変換できます。通常はGUIノードを使ってください。

## RTX 4070での実測

### 2048トークン分割への更新 — 2026年10月3日

同じRTX 4070 12 GBで、4ステップ・544 × 960のFL2VA Case 01を各方式3回ずつ比較しました。実行時間の中央値は8分割の**60.319秒**から2048トークン自動分割の**59.212秒**へ短縮しました。samplerの中央値は38.700秒から37.509秒へ短縮。自動分割の各実行は59.212／59.284／58.884秒でした。各比較内ではモデル・プロンプト・seed・ステップ数・Attention設定を揃えています。

さらに、画像・プロンプトを変えた先頭フレーム条件付けの3ケースを、124フレーム・24 fps・生成音声付きで比較しました。

| 生成解像度 → X2出力 | 8分割・秒 | 2048自動分割・秒 |
|---|---:|---:|
| 512 × 512 → 1024 × 1024 | 50.491 | 48.951 |
| 768 × 512 → 1536 × 1024 | 49.856 | 49.152 |
| 768 × 1024 → 1536 × 2048 | 83.444 | 82.295 |

追加解像度は**各方式1回**のため、差は参考値であり、反復測定による推定ではありません。計測は`execution_start`から`execution_success`までで、デコード・保存を含み、サーバー起動・メディア検査を除きます。実行キャッシュのヒットはありませんが、OS／コンパイラのキャッシュや初期化の影響は分離していません。6本すべてOOMなしで生成し、全編のデコード検査を通過しました。正方形のペアはデコードRGB・音声PCMが完全一致しましたが、残りのペアには差があり、ビット一致を保証するものではありません。SLAはkeep 5%・`min_tokens=12288`で、正方形ケースは閾値未満のため両方式ともDense Attentionです。

### 初回のストリーミング実測 — 2026年10月2日

**2026年10月2日**、Windows、RTX 4070 12 GBで計測しました。seed 43、124フレーム／24 fps（約5.17秒）、モデル生成音声付き。生成解像度544 × 960、X2出力1088 × 1920です。比較元は[さやかベンチマーク](https://sayakabenchmark.pages.dev/)のFL2VA `017-matlow-fused4-sla5-x2vae`です。

| 構成 | Case 01・秒 | Case 02・秒 | 60秒未満 |
|---|---:|---:|---:|
| 基準：FP16 X2、FFN 4分割 | 86.725（1回） | 82.680（1回） | 0/2 |
| **4ステップ、INT8 X2、FFN 8分割、非同期NVENC** | 59.006 / 59.794 / 60.300、**平均59.700** | 59.634 / 59.783 / 59.962、**平均59.793** | **5/6** |
| 3ステップ実験 | 50.918 / 51.864 / 52.007、平均51.596 | 51.387 / 51.943 / 51.737、平均51.689 | 6/6 |

この表は**開発時の試作コードでの実測**です。記録は[`docs/benchmark.json`](docs/benchmark.json)、公開用コードの確認結果は別途[`docs/validation.md`](docs/validation.md)に掲載しています。計測範囲はサーバー起動後のComfyUI `execution_start`から`execution_success`まで。グラフキャッシュなし（`--cache-none`）、pinned memoryなし（`--disable-pinned-memory`）です。サーバー起動、初回INT8変換、生成後のメディア検査は含みません。アプリ起動やダウンロードからの所要時間ではありません。

samplerは`res_multistep`／`simple`、BasicGuider、video/audio shift 12/3、SLA 5%、`min_tokens=12288`、`extra_tokens=256`、条件付けと音声の行はexactにする設定です。FP16 accumulationはVAEデコード中だけ有効にして元に戻します。NVENCはH.264、p4/hq、VBR CQ18、yuv420p、sRGB transfer／BT.709 matrix・primaries、音声はAACです。

**配布する標準は4ステップです。** 3ステップ版ではCase 02の黒い袖が白くなる差が見られたため、推奨設定にはせず、実用ワークフローも同梱していません。

INT8変換対象はdecoder block内のLinear重み144個だけです。per-channel INT8 ConvRot・group size 256を使い、encoder、norm、bias、最終X2 projectionのFP16を保ちます。開発時に同じlatentの圧縮前7フレームずつを比較した平均PSNRは57.75／57.83 dBでした。同期保存と非同期保存の比較では124フレームのデコードRGBと音声PCMが一致しました。全プロンプトで見た目・音が同等になる保証ではなく、最終的な品質判断は人による確認が必要です。

## 困ったとき・削除方法

- **H3やSparse Attentionノードがない：** 確認済みComfyUI版か、必要機能を含む新しい版を使ってください。付属例のSparse AttentionはComfyUI本体のノードです。
- **CUDA／INT8カーネルのエラー：** 使用中のComfyUI専用Python、CUDA版PyTorch、Comfy Kitchenの組み合わせを確認してください。CPU向けの代替経路はありません。
- **NVENCのエラー：** GPU番号、ドライバーのH.264 NVENC対応、PyAV側のエンコーダーを確認してください。ソフトウェアエンコードへ自動変更はしません。
- **元モデルのhash不一致：** 対応対象は元のX2 Detail v1のみです。SHA-256は`2296840f4acedcaa976688e7d7b97f7bf570b136e400385d3f46224011897aac`です。
- **既存の変換先が不適合：** 既存ファイルは残して別の出力名を指定してください。過去の非公開実験で作ったモデルには公開用converterの識別情報がありません。
- **変換中にプロセスが異常終了：** `models/vae/h3_x2_stream`に`.lock`や`.h3-x2-*`一時ファイルが残る場合があります。変換プロセスが動いていないことを確認し、該当する残留ファイルだけを除いて再実行してください。通常のエラー／中断では自動清掃します。
- **VRAM不足：** ほかのGPU処理を終えるか、生成サイズを下げてください。H3のデコード経路を保つため、一般的な3D tiled OOM fallbackへは切り替えません。
- 削除する場合はComfyUIを停止してカスタムノードのフォルダを削除します。変換済みモデルは`models/vae/h3_x2_stream`に別保存されるので、必要に応じて明示的に残す／削除してください。グローバル設定の復元は不要です。

## 謝辞とライセンス

コードはGPL-3.0です。[LICENSE](LICENSE)と[NOTICE.md](NOTICE.md)にKJNodes由来部分、ComfyUI／Comfy Kitchen、X2モデルの作者と利用条件を記載しています。変換済みの重みも元モデルの利用条件に従います。コードのライセンスによって重みの権利が追加されることはありません。
