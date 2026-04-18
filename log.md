# 3DGS 部屋アーカイブ プロジェクト ログ

## 2026-04-18

### 開始
- specification.md を確認し、3DGS パイプラインの実装を開始
- 対象: 静止した室内空間の3Dアーカイブ
- 動作環境: NVIDIA RTX 3050 (VRAM 8GB)
- パイプライン: 前処理 → COLMAP → 学習データ準備 → 3DGS学習 → レンダリング

### 実装計画
1. config.yaml - 設定ファイル
2. requirements.txt - 依存パッケージ
3. src/utils/ - ユーティリティモジュール
4. src/preprocess.py - 画像前処理
5. src/run_colmap.py - COLMAP実行
6. src/prepare_training.py - 学習データ準備
7. src/train.py - 3DGS学習
8. src/render.py - レンダリング・品質評価

### 実装完了

#### ディレクトリ構成
- `img/frames/` - 入力画像ディレクトリ（空）
- `data/processed/` - 前処理済み画像出力先
- `data/colmap/sparse/` - COLMAP 出力先
- `data/training/` - 3DGS 学習データ
- `output/models/` - 学習済みモデル出力先
- `output/renders/` - レンダリング結果出力先

#### 作成ファイル一覧
| ファイル | 内容 |
|---------|------|
| `config.yaml` | RTX 3050 8GB 向けに最適化された設定（sh_degree:1, resolution:2） |
| `requirements.txt` | PyYAML, Pillow, numpy, opencv-python, tqdm |
| `src/utils/image_utils.py` | ブレ検出（ラプラシアン分散）、リサイズ、画像ファイル取得 |
| `src/utils/colmap_utils.py` | COLMAP コマンド実行、インストール確認、登録画像数カウント |
| `src/preprocess.py` | 画像前処理（リサイズ・ブレ除去・連番リネーム） |
| `src/run_colmap.py` | COLMAP 4ステップ実行（特徴抽出→マッチング→SfM→変換） |
| `src/prepare_training.py` | COLMAP出力を3DGS形式に整形（シンボリックリンク活用） |
| `src/train.py` | 3DGS公式train.pyのラッパー（config.yamlからパラメータ注入） |
| `src/render.py` | レンダリング＋品質評価（PSNR/SSIM） |

#### RTX 3050 8GB 向け OOM 対策（config.yaml に反映済み）
- `resolution: 2` → 学習解像度を1/2に縮小
- `sh_degree: 1` → SH次数を下げてVRAM節約
- `densify_until_iter: 10000` → Densification を早期終了
- `max_long_side: 1200` → 前処理で画像を1200px以下にリサイズ

### 次のステップ（実行手順）
```bash
# 0. 環境セットアップ
git clone https://github.com/graphdeco-inria/gaussian-splatting --recursive
pip install -r requirements.txt

# 1. img/frames/ に撮影画像を配置

# 2. 画像前処理
python src/preprocess.py --config config.yaml

# 3. COLMAP によるカメラ推定
python src/run_colmap.py --config config.yaml

# 4. 学習データ準備
python src/prepare_training.py --config config.yaml

# 5. 3DGS 学習（GPU 必須、3〜5時間）
python src/train.py --config config.yaml

# 6. レンダリング・品質確認
python src/render.py --config config.yaml
```

### gaussian-splatting をサブモジュールとして追加

- `.gitignore` から `gaussian-splatting/` の除外を解除
- `git submodule add` でサブモジュール登録
- `git submodule update --init --recursive` でサブモジュール（SIBR_viewers, diff-gaussian-rasterization, fused-ssim, simple-knn, glm）を初期化

### Python 3.9 互換性修正

- `str | None` (Python 3.10+) → `Optional[str]` に修正
- `list[str]` → `List[str]` に修正
- `from __future__ import annotations` を追加
- 対象: `colmap_utils.py`, `image_utils.py`, `render.py`

### カメラモデル修正: OPENCV → PINHOLE

- 3DGS 公式実装は `PINHOLE` モデルのみ対応（`assert model == "PINHOLE"` でチェックされる）
- `config.yaml` の `camera_model` を `OPENCV` → `PINHOLE` に変更
- COLMAP の再実行が必要（Step 2〜3 をやり直し）

### 3DGS 学習完了

- 30,000 イテレーションの学習が正常に完了
- 出力モデル: `output/models/room_archive/point_cloud/iteration_30000/point_cloud.ply`
- 3D表示: https://supersplat.io に .ply をドラッグ&ドロップで確認可能
