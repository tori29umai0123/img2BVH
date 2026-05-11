# image2BVH

Windows / Linux / macOS 対応のシンプルな WebUI。1枚の画像から人物を
SAM 3 で抽出し、人数分のポーズ推定→人ごとの BVH を出力します。
本プロジェクトは下記 2 つの ComfyUI カスタムノードを参考にした
スタンドアロン実装です（[クレジット](#クレジット--参考リポジトリ)参照）。

## パイプライン

1. 画像入力（PNG / JPG）
2. **SAM 3** で `person`（固定）に一致するインスタンスを idx 別に抽出
3. UI から不要な人物をスキップ・任意で人物ごとの左右手画像で hand pose を上書き
4. **SAM 3D Body** で各人のポーズを推定（`pose_adjust` スライダーで前のめり補正）
5. **純 Python の BVH ライタ** で人数分の BVH を `tmp/` に書き出し → ブラウザから個別 DL（Blender 不要）

## 対応プラットフォーム

| OS / arch | torch |
| --- | --- |
| Windows x64 (AMD64)         | `cu130` nightly |
| Linux x86_64                | `cu130` nightly + 専用 triton |
| Linux aarch64 (DGX Spark)   | `cu130` nightly + 専用 triton |
| macOS arm64 (Apple Silicon) | CPU / MPS (PyPI 標準) |
| macOS x64 (Intel)           | CPU (PyPI 標準) |

CUDA 系は全プラットフォームで cu130 nightly に統一しています。理由は
DGX Spark (GB10 / sm_121 = Blackwell Ultra) を動かすために cu130 が必須で
（CUDA 12.8 stable の NVRTC は sm_120 までしか JIT コンパイル不可）、
他環境と toolchain バージョンを揃えて実機間の挙動差を抑えるためです。
分岐は `pyproject.toml` の `[tool.uv.sources]` と `marker` で自動判定されます。
nightly のため日付ピン (`2.13.0.dev20260422+cu130`) を指定しています。

## 必要なもの

- Python 3.11（無ければ run スクリプトが uv 経由で自動インストール）
- `uv`（PATH に無ければ run スクリプトが自動でインストール）
- **HuggingFace アカウント＋ SAM 3 へのアクセス承認**（後述）

## 事前準備：SAM 3 へのアクセス申請（必須）

`facebook/sam3` は Meta による **manual gating** リポジトリです。
初回 DL の前に下記 2 ステップが必要です（1 度だけ）。

1. https://huggingface.co/facebook/sam3 を開いて **「Agree and access repository」**
   をクリック。氏名・生年月日・国・所属・職種などを記入して申請
   - 自動承認ではありません（Meta による手動審査、通常 数時間〜数日）
2. https://huggingface.co/settings/tokens で **read** スコープのアクセストークンを発行

トークンの渡し方（次のいずれか）：

- **`run.bat` / `run.sh` の自動プロンプト（推奨・最も簡単）**
  - 起動時に未ログインを自動検知し、トークン入力欄を表示
  - 入力されたトークンは内部で `huggingface-cli login --token` にパイプされ、
    HF 標準のトークンファイル（`~/.cache/huggingface/token`）に保存
  - 一度入力すれば次回以降は省略
  - 承認待ちの間でも Enter でスキップ可能（後で再起動して入れ直せる）
- 環境変数： 起動前に `HF_TOKEN=hf_xxxxxxxx` をセットしておけば、プロンプトはスキップされる
- 手動： `uv run --no-dev huggingface-cli login` を 1 回叩いておく

承認前に「マスク生成」ボタンを押すと、ログに以下のような案内が表示されます。
申請＆トークン投入後にもう一度押せば DL が走ります（既に DL 済みのファイルは保持）。

```
[SAM3] HuggingFace repo 'facebook/sam3' is GATED (manual approval by Meta).
  1) Open https://huggingface.co/facebook/sam3 and click 'Request access'.
  ...
```

なお SAM 3D Body（`jetjodh/sam-3d-body-dinov3`）はコミュニティミラーで
gating なし、Blender Portable も gating とは無関係です。
URL / repo_id は `config.ini` から差し替えられます（公式 SAM3 が嫌なら
非 gated なミラーに変更可能）。

## 起動

### Windows

```cmd
run.bat
```

### Linux / macOS

```bash
./run.sh
```

### 共通の流れ

1. `uv sync --no-dev` で `.venv` 構築 ＋ 依存解決（platform に応じて
   `cu130 nightly` / CPU 自動選択）
2. WebUI 起動 → ブラウザで `http://127.0.0.1:7860`
3. **Step 1**: 画像をアップロードして「マスク生成」→ 必要なら不要な人物をスキップ
4. **Step 2** (任意): 人物ごとに左/右手画像 + 左右反転を設定
5. **Step 3**: 「ポーズ推定 → BVH 出力」
6. `tmp/person_NN.bvh` がブラウザから個別 DL 可能（推論ごとに `tmp/` は全削除されます）

HuggingFace のモデル重みは無い場合のみ自動 DL（冪等）：
- SAM 3 ウェイト → `runtime/models/sam3/`
- SAM 3D Body ウェイト → `runtime/models/sam3dbody/`

HuggingFace の repo_id は `config.ini`（初回起動時に自動生成）から編集できます。

### BVH 書き出し

BVH は `image2bvh/bvh_writer.py` の純 Python ライタで書き出します。Blender 等の
外部ツールに依存しません。HIERARCHY OFFSET にポーズを焼き込み、MOTION の回転
チャンネルはすべてゼロという「静止 1 フレーム」スタイルです。

> 旧版では Blender Portable 経由の BVH/FBX 書き出しもサポートしていましたが、
> 起動時の DL 負荷と運用上の脆弱性を理由に廃止しました。`runtime/blender/` が
> 残っている場合は安全に手動削除できます（参照されません）。

## ディレクトリ構成

```
image2BVH/
├── run.bat                 # Windows ランチャ
├── run.sh                  # Linux / macOS ランチャ
├── pyproject.toml          # uv 管理 / platform marker で torch を分岐
├── config.ini              # 言語・モデル repo_id・log表示の設定（自動生成）
├── image2bvh/
│   ├── server.py           # FastAPI / uvicorn エントリポイント
│   ├── bootstrap.py        # SAM3 / SAM 3D Body の HF Hub DL（config.ini 駆動）
│   ├── config.py           # config.ini 読み書き
│   ├── segmentation.py     # SAM3 keyword 抽出 + 色パレット
│   ├── pose.py             # SAM3D Body ポーズ推定（hand override + lean 補正）
│   ├── hand_inference.py   # 手画像 → hand_pose_params 上書き
│   ├── bvh_writer.py       # 純 Python BVH ライタ（HIERARCHY + 1F MOTION）
│   ├── bvh_export.py       # PersonPose のリストを BVH ファイル群に書き出し
│   ├── vram.py             # VRAM 計測ヘルパ
│   ├── paths.py
│   └── vendor/sam_3d_body/ # 参考リポジトリより同梱
├── web/                    # 自前 WebUI (HTML / CSS / JS / i18n / preview3d)
├── runtime/                # 初回 DL 物 (.gitignore 済)
│   └── models/{sam3, sam3dbody}/
└── tmp/                    # BVH 出力先 (推論ごとにクリア → ブラウザでDL)
```

## トラブルシュート

- **DGX Spark で torch が落ちる**: `nvidia-smi` で sm_121 が見えていることを確認。
  `pyproject.toml` の cu130 nightly index が反映されているか `uv sync` ログを確認。
- **既存の `.venv` から cu128 → cu130 に切り替えたい**: ルートの `uv.lock` と
  `.venv/` を削除してから `run.bat` / `run.sh` を再実行。新しい cu130 nightly
  で lock が再生成されます。
- **SAM3 / SAM3D Body の HF ダウンロードが遅い / 失敗**: 環境変数
  `HF_HUB_ENABLE_HF_TRANSFER=1` を立ててから run スクリプトを起動すると高速化。

## クレジット / 参考リポジトリ

本プロジェクトは下記 2 つの ComfyUI カスタムノードのコード・モデルロード手順
を参考にしたスタンドアロン WebUI 実装です。
クレジットおよび上流コードのライセンスは各リポジトリに従います。

- **SAM 3（テキストプロンプトによるマルチインスタンスセグメンテーション）**
  - [wouterverweirder/comfyui_sam3](https://github.com/wouterverweirder/comfyui_sam3)
- **SAM 3D Body（人物 3D ポーズ／メッシュ推定）**
  - [PozzettiAndrea/ComfyUI-SAM3DBody](https://github.com/PozzettiAndrea/ComfyUI-SAM3DBody)
  - 同リポジトリの `nodes/sam_3d_body/` を `image2bvh/vendor/sam_3d_body/` として
    同梱し、本プロジェクトのインターフェース層から呼び出しています。BVH 書き出しは
    純 Python 実装（`image2bvh/bvh_writer.py`）に置き換えています。
- **前のめり補正アルゴリズム**
  - [tori29umai0123/ComfyUI-SAM3DBody_utills](https://github.com/tori29umai0123/ComfyUI-SAM3DBody_utills)
  - `nodes/processing/process.py:apply_pose_lean_correction_rig` のロジックを参考に
    `image2bvh/pose.py:apply_lean_correction` を実装。spine/neck/head のチェーンを
    背側に反らせて SAM 3D Body の前傾バイアスを補正。

## ライセンス

本リポジトリは **複数のライセンス** が混在しています。配布・改変・二次利用の際は
それぞれの条件を確認してください。

| 対象 | ライセンス | テキスト |
| --- | --- | --- |
| 自作部分（`image2bvh/*.py`（vendor サブツリー除く）／`web/`／`run.bat`／`run.sh`／`pyproject.toml` ほか） | **MIT** | [`LICENSE`](LICENSE) |
| `image2bvh/vendor/sam_3d_body/`（SAM 3D Body 推論コード一式） | **Meta SAM License** | [`LICENSES/SAM_LICENSE.txt`](LICENSES/SAM_LICENSE.txt) |
| `image2bvh/vendor/sam_3d_body/models/backbones/dinov3_repo/`（DINOv3 ソース同梱） | **Meta DINOv3 License** | [`LICENSES/DINOv3_LICENSE.md`](LICENSES/DINOv3_LICENSE.md) |
| `runtime/models/sam3/`（実行時 DL される SAM 3 重み） | Meta SAM License（リポジトリ非同梱） | リポジトリには含めず、各ユーザが HF ハブから取得 |
| `runtime/models/sam3dbody/`（実行時 DL される SAM 3D Body 重み） | Meta SAM License（リポジトリ非同梱） | 同上 |

帰属表示の詳細は [`NOTICE`](NOTICE) を参照してください。

### 再配布する場合の重要事項

- SAM License の `SAM Materials` 定義には **推論コードも含まれる** ため、`image2bvh/vendor/sam_3d_body/` を
  含めて再配布する際は SAM License の条項下で配布する必要があります（同ライセンステキストの同梱必須）。
- DINOv3 License も同等の継承義務を持ちます（`dinov3_repo/` 同梱時に該当）。
- SAM / DINOv3 ライセンスは **軍事・武器・核・諜報用途・ITAR 規制用途を禁止** しています。
- 上記は理解しやすさのための要約であり、法的助言ではありません。商用利用・大規模配布の際は
  知財の専門家に相談することを強く推奨します。
