# 3DGS 部屋アーカイブ プロジェクト仕様書

**バージョン:** 1.1.0  
**作成日:** 2026-04-18  
**ステータス:** Draft  
**動作環境:** NVIDIA RTX 3050（VRAM 8GB）

---

## 目次

1. [プロジェクト概要](#1-プロジェクト概要)
2. [技術スタック](#2-技術スタック)
3. [ディレクトリ構成](#3-ディレクトリ構成)
4. [処理パイプライン](#4-処理パイプライン)
5. [各モジュール仕様](#5-各モジュール仕様)
6. [入出力仕様](#6-入出力仕様)
7. [実行手順](#7-実行手順)
8. [品質基準](#8-品質基準)
9. [制約・前提条件](#9-制約前提条件)

---

## 1. プロジェクト概要

### 1.1 目的

3D Gaussian Splatting（3DGS）を用いて、居住空間（自室）を高品質な3Dモデルとしてアーカイブする。撮影した静止画像群から、リアルタイムレンダリング可能な3Dシーンを生成・保存することを目的とする。

### 1.2 成果物

| 成果物 | 説明 |
|--------|------|
| 学習済み 3DGS モデル | `.ply` 形式のガウシアンスプラットデータ |
| カメラ姿勢推定結果 | COLMAP による SfM 出力 |
| レンダリング済み画像 | 任意視点からのレンダリング結果 |
| インタラクティブビューア | ブラウザ上で閲覧可能な Web ビューア（任意） |

### 1.3 スコープ

- **対象:** 静止した室内空間（撮影時に変化しない前提）
- **入力:** `img/frames/` 内の静止画像 約400枚
- **出力:** 3DGS 形式の3Dシーンファイル

---

## 2. 技術スタック

| カテゴリ | ツール / ライブラリ | バージョン目安 | 役割 |
|----------|---------------------|----------------|------|
| 3D再構成 | [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) | 公式実装 | メインの学習・レンダリング |
| SfM | COLMAP | 3.8+ | カメラ姿勢推定・疎な点群生成 |
| 画像処理 | FFmpeg / Python Pillow | - | フレーム前処理 |
| 言語 | Python | 3.10+ | パイプライン自動化スクリプト |
| GPU | CUDA 対応 GPU | CUDA 11.8+ | 3DGS 学習の実行 |
| ビューア | SuperSplat / 3DGS Web Viewer | - | 結果確認（任意） |

---

## 3. ディレクトリ構成

```
project_root/
│
├── img/
│   └── frames/                  # 入力画像（約400枚）
│       ├── frame_0001.jpg
│       ├── frame_0002.jpg
│       └── ...
│
├── src/
│   ├── preprocess.py            # 画像前処理スクリプト
│   ├── run_colmap.py            # COLMAP 実行スクリプト
│   ├── prepare_training.py      # 3DGS 学習データ準備
│   ├── train.py                 # 3DGS 学習実行スクリプト
│   ├── render.py                # レンダリングスクリプト
│   └── utils/
│       ├── image_utils.py       # 画像ユーティリティ
│       └── colmap_utils.py      # COLMAP ユーティリティ
│
├── data/
│   ├── colmap/                  # COLMAP の出力先
│   │   ├── sparse/              # 疎な点群・カメラ情報
│   │   └── dense/               # 密な点群（任意）
│   └── processed/               # 前処理済み画像
│
├── output/
│   ├── models/                  # 学習済み 3DGS モデル (.ply)
│   └── renders/                 # レンダリング済み画像
│
├── gaussian-splatting/          # 3DGS 公式実装（サブモジュールまたはクローン）
│
├── requirements.txt             # Python 依存パッケージ
├── config.yaml                  # 設定ファイル
└── README.md
```

---

## 4. 処理パイプライン

```
[入力画像 img/frames/]
        │
        ▼
[Step 1] 画像前処理 (preprocess.py)
  - 解像度統一
  - 露出補正（任意）
  - ブレ・低品質フレームの除去
        │
        ▼
[Step 2] カメラ姿勢推定 (run_colmap.py)
  - Feature Extraction
  - Feature Matching
  - Sparse Reconstruction (SfM)
        │
        ▼
[Step 3] 学習データ準備 (prepare_training.py)
  - COLMAP 出力を 3DGS 入力形式に変換
  - images/ と sparse/ の配置
        │
        ▼
[Step 4] 3DGS 学習 (train.py)
  - ガウシアン初期化
  - 反復最適化（デフォルト 30,000 イテレーション）
  - モデル保存 (.ply)
        │
        ▼
[Step 5] レンダリング・確認 (render.py)
  - 任意視点からのレンダリング
  - 品質評価（PSNR / SSIM）
        │
        ▼
[成果物] output/models/*.ply
```

---

## 5. 各モジュール仕様

### 5.1 `src/preprocess.py` — 画像前処理

**目的:** 入力画像を COLMAP および 3DGS 学習に適した形式に整える。

**処理内容:**

| 処理 | 詳細 |
|------|------|
| 解像度変換 | 長辺を 1600px 以下にリサイズ（メモリ制約に応じて調整） |
| 形式統一 | すべて JPEG または PNG に統一 |
| ファイル連番リネーム | `frame_XXXX.jpg` 形式に統一 |
| 低品質フレームの除去 | ブレ検出（ラプラシアン分散）によるフィルタリング |

**入力:** `img/frames/*.jpg`（または各種形式）  
**出力:** `data/processed/*.jpg`

**主要パラメータ（config.yaml より読み込み）:**

```yaml
preprocess:
  max_long_side: 1200       # リサイズ上限（px）※RTX 3050 8GB では 1600 だと OOM リスクあり
  blur_threshold: 100.0     # ラプラシアン分散の閾値（小さいほど除去）
  output_format: "jpg"
```

---

### 5.2 `src/run_colmap.py` — COLMAP 実行

**目的:** 前処理済み画像群からカメラ姿勢と疎な点群を推定する。

**処理ステップ:**

1. `colmap feature_extractor` — 各画像から特徴点を抽出
2. `colmap exhaustive_matcher` / `sequential_matcher` — 特徴点マッチング
3. `colmap mapper` — バンドル調整による SfM
4. `colmap model_converter` — テキスト形式への変換

**入力:** `data/processed/`  
**出力:** `data/colmap/sparse/0/`（`cameras.txt`, `images.txt`, `points3D.txt`）

**主要パラメータ:**

```yaml
colmap:
  matcher: "exhaustive"     # 'exhaustive' or 'sequential'（動画フレームは sequential が速い）
  use_gpu: true
  camera_model: "OPENCV"    # カメラモデル（歪み補正あり）
  single_camera: true       # 全画像を同一カメラとして扱う
```

---

### 5.3 `src/prepare_training.py` — 学習データ準備

**目的:** COLMAP 出力を 3DGS 公式実装が要求する形式に整形する。

**処理内容:**
- 画像ディレクトリと sparse ディレクトリの配置確認
- 3DGS の `convert.py`（公式実装内）の呼び出し
- カメラパラメータの整合性チェック

**入力:** `data/colmap/`, `data/processed/`  
**出力:** `data/training/`（3DGS 学習入力形式）

---

### 5.4 `src/train.py` — 3DGS 学習実行

**目的:** 準備されたデータを用いて 3D Gaussian Splatting の学習を実行する。

**処理内容:**
- 3DGS 公式 `train.py` をラップして実行
- 設定ファイルからハイパーパラメータを注入
- 学習ログの保存

**入力:** `data/training/`  
**出力:** `output/models/<experiment_name>/`（`.ply` ファイル群）

**主要パラメータ:**

```yaml
training:
  iterations: 30000                  # 学習イテレーション数
  save_iterations: [7000, 30000]     # 中間保存タイミング
  test_iterations: [7000, 30000]     # 評価タイミング
  densify_until_iter: 10000          # Densification 終了イテレーション ※8GBでガウシアン爆増を抑制するため 15000→10000
  densify_from_iter: 500
  densification_interval: 100
  opacity_reset_interval: 3000
  position_lr_init: 0.00016
  position_lr_final: 0.0000016
  feature_lr: 0.0025
  sh_degree: 1                       # ※8GB VRAM では 3 だと OOM になりやすいため 1 に設定（品質は若干低下）
  white_background: false            # 部屋撮影は false
  resolution: 2                      # ※学習時の内部解像度を 1/2 にダウンスケール（VRAM 節約）
```

> **⚠️ RTX 3050 8GB 向けメモ:**  
> `sh_degree: 3` はガウシアン 1 点あたりのパラメータが約 4 倍になり VRAM を大きく消費する。  
> まず `sh_degree: 1` で動作確認し、余裕があれば `2` に上げる。  
> `resolution: 2` は学習画像を半分サイズに縮小して使うオプション（前処理リサイズとは別）。  
> OOM が発生する場合は `densify_until_iter` をさらに `8000` まで下げることを検討。

---

### 5.5 `src/render.py` — レンダリング

**目的:** 学習済みモデルから任意視点の画像を生成・品質評価する。

**処理内容:**
- 3DGS 公式 `render.py` をラップして実行
- 学習時と同じカメラ視点での評価レンダリング
- PSNR / SSIM の計算と集計

**入力:** `output/models/<experiment_name>/`  
**出力:** `output/renders/<experiment_name>/`

---

## 6. 入出力仕様

### 6.1 入力画像仕様

| 項目 | 仕様 |
|------|------|
| 枚数 | 約400枚 |
| 形式 | JPEG / PNG |
| 解像度 | 任意（前処理でリサイズ） |
| 推奨撮影方法 | 部屋を周回しながら全方向をカバー。重複率 60〜80% を確保 |
| 注意 | 撮影中に物が移動しないこと。人物・反射面に注意 |

### 6.2 出力ファイル仕様

| ファイル | 形式 | 説明 |
|----------|------|------|
| `point_cloud/iteration_30000/point_cloud.ply` | PLY | 最終モデル（3DGS 点群） |
| `cameras.json` | JSON | カメラ情報 |
| `cfg_args` | テキスト | 学習時の引数 |

---

## 7. 実行手順

```bash
# 0. 環境セットアップ
git clone https://github.com/graphdeco-inria/gaussian-splatting --recursive
pip install -r requirements.txt

# 1. 画像前処理
python src/preprocess.py --config config.yaml

# 2. COLMAP によるカメラ推定
python src/run_colmap.py --config config.yaml

# 3. 学習データ準備
python src/prepare_training.py --config config.yaml

# 4. 3DGS 学習（GPU 必須）
python src/train.py --config config.yaml

# 5. レンダリング・品質確認
python src/render.py --config config.yaml \
  --model_path output/models/<experiment_name>
```

---

## 8. 品質基準

| 指標 | 目標値 | 備考 |
|------|--------|------|
| PSNR | 28 dB 以上 | 学習視点での評価 |
| SSIM | 0.85 以上 | 学習視点での評価 |
| COLMAP 登録画像数 | 入力の 80% 以上 | 少ない場合は撮影方法を見直し |
| 学習時間 | 3〜5時間（30k iter） | RTX 3050 8GB 基準 |

---

## 9. 制約・前提条件

### 9.1 ハードウェア要件

| 項目 | 本プロジェクト環境 | 備考 |
|------|----------|------|
| GPU | NVIDIA RTX 3050（VRAM 8GB） | CUDA 11.8 以上が必要 |
| RAM | 16GB 以上推奨 | COLMAP の Exhaustive Matching は RAM を多く消費 |
| ストレージ | 50GB 以上の空き | SSD 推奨 |
| OS | Linux / Windows（WSL2） | Ubuntu 22.04 推奨 |

### 9.2 前提条件

- CUDA Toolkit がインストール済みであること
- COLMAP がインストール済み（または conda 環境）であること
- 撮影画像に移動物体（人・ペット等）が映り込んでいないこと
- 撮影時の照明条件が安定していること（窓からの自然光変化に注意）

### 9.3 既知の制限事項

- ガラス・鏡・光沢面は 3DGS が苦手とする素材であり、再現精度が低下する場合がある
- COLMAP が失敗する場合、特徴点の少ない画像（真っ白な壁のみ等）が原因のことが多い
- **RTX 3050 8GB での OOM 対策（優先度順）:**
  1. `resolution: 2` → 学習解像度を 1/2 に（最優先・効果大）
  2. `sh_degree: 1` → SH 次数を下げる（デフォルト設定済み）
  3. `densify_until_iter: 8000` → Densification をさらに早期終了
  4. `max_long_side: 800` → 前処理解像度をさらに下げる
  5. 上記すべて適用後も OOM の場合、画像枚数を 200 枚程度に間引く

---