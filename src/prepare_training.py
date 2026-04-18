"""学習データ準備スクリプト

COLMAP 出力を 3DGS 公式実装が要求する形式に整形する。

3DGS が要求するディレクトリ構成:
  data/training/
  ├── images/        # 学習用画像
  └── sparse/0/      # COLMAP 疎な点群・カメラ情報
      ├── cameras.bin (or .txt)
      ├── images.bin (or .txt)
      └── points3D.bin (or .txt)
"""

import argparse
import shutil
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils.colmap_utils import count_registered_images


def prepare_training(config_path: str) -> None:
    with open(config_path) as f:
        config = yaml.safe_load(f)

    root = Path(config["project"]["root"])
    processed_dir = root / config["paths"]["processed"]
    colmap_dir = root / config["paths"]["colmap_output"]
    training_dir = root / config["paths"]["training_data"]
    sparse_src = colmap_dir / "sparse" / "0"

    # 入力チェック
    if not sparse_src.exists():
        print(f"エラー: COLMAP の疎な点群が見つかりません: {sparse_src}")
        print("先に run_colmap.py を実行してください。")
        sys.exit(1)

    if not processed_dir.exists():
        print(f"エラー: 前処理済み画像が見つかりません: {processed_dir}")
        sys.exit(1)

    # 3DGS 用ディレクトリ作成
    training_images = training_dir / "images"
    training_sparse = training_dir / "sparse" / "0"

    training_images.mkdir(parents=True, exist_ok=True)
    training_sparse.mkdir(parents=True, exist_ok=True)

    # 画像のコピー（シンボリックリンクで容量節約）
    print("画像をリンク中...")
    image_count = 0
    for img in sorted(processed_dir.iterdir()):
        if img.suffix.lower() in (".jpg", ".jpeg", ".png"):
            dest = training_images / img.name
            if dest.exists() or dest.is_symlink():
                dest.unlink()
            dest.symlink_to(img.resolve())
            image_count += 1
    print(f"  {image_count} 枚の画像をリンクしました")

    # COLMAP sparse データのコピー
    print("COLMAP データをコピー中...")
    required_files = ["cameras.txt", "images.txt", "points3D.txt"]
    bin_files = ["cameras.bin", "images.bin", "points3D.bin"]

    # テキスト形式を優先、なければバイナリ形式をコピー
    copied = False
    for files in [required_files, bin_files]:
        if all((sparse_src / f).exists() for f in files):
            for f in files:
                shutil.copy2(sparse_src / f, training_sparse / f)
            copied = True
            print(f"  {', '.join(files)} をコピーしました")
            break

    if not copied:
        print("エラー: COLMAP の出力ファイルが見つかりません。")
        print(f"  確認先: {sparse_src}")
        sys.exit(1)

    # カメラパラメータの整合性チェック
    registered = count_registered_images(str(training_sparse))
    print(f"\n学習データ準備完了:")
    print(f"  画像数: {image_count}")
    print(f"  COLMAP 登録画像数: {registered}")
    print(f"  出力先: {training_dir}")

    if registered == 0:
        print("  警告: 登録画像数が 0 です。COLMAP の結果を確認してください。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3DGS 学習データ準備")
    parser.add_argument("--config", default="config.yaml", help="設定ファイルのパス")
    args = parser.parse_args()
    prepare_training(args.config)
