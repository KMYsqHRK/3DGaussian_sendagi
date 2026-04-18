"""COLMAP 実行スクリプト

前処理済み画像群からカメラ姿勢と疎な点群を推定する。
1. feature_extractor — 各画像から特徴点を抽出
2. exhaustive_matcher / sequential_matcher — 特徴点マッチング
3. mapper — バンドル調整による SfM
4. model_converter — テキスト形式への変換
"""

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils.colmap_utils import check_colmap_installed, run_colmap_command, count_registered_images
from src.utils.image_utils import get_image_files


def run_colmap(config_path: str) -> None:
    with open(config_path) as f:
        config = yaml.safe_load(f)

    root = Path(config["project"]["root"])
    image_dir = root / config["paths"]["processed"]
    colmap_dir = root / config["paths"]["colmap_output"]
    sparse_dir = colmap_dir / "sparse"
    database_path = colmap_dir / "database.db"
    log_file = str(colmap_dir / "colmap.log")

    colmap_cfg = config["colmap"]
    matcher = colmap_cfg["matcher"]
    use_gpu = colmap_cfg["use_gpu"]
    camera_model = colmap_cfg["camera_model"]
    single_camera = colmap_cfg["single_camera"]

    if not check_colmap_installed():
        print("エラー: COLMAP がインストールされていません。")
        sys.exit(1)

    # 入力画像数を確認
    input_images = get_image_files(str(image_dir))
    if not input_images:
        print(f"エラー: 前処理済み画像が見つかりません: {image_dir}")
        print("先に preprocess.py を実行してください。")
        sys.exit(1)
    total_images = len(input_images)
    print(f"入力画像数: {total_images}")

    colmap_dir.mkdir(parents=True, exist_ok=True)
    sparse_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Feature Extraction
    print("\n=== Step 1: Feature Extraction ===")
    fe_args = [
        "feature_extractor",
        "--database_path", str(database_path),
        "--image_path", str(image_dir),
        "--ImageReader.camera_model", camera_model,
        "--SiftExtraction.use_gpu", "1" if use_gpu else "0",
    ]
    if single_camera:
        fe_args += ["--ImageReader.single_camera", "1"]
    run_colmap_command(fe_args, log_file)
    print("Feature Extraction 完了")

    # Step 2: Feature Matching
    print(f"\n=== Step 2: Feature Matching ({matcher}) ===")
    if matcher == "exhaustive":
        match_args = [
            "exhaustive_matcher",
            "--database_path", str(database_path),
            "--SiftMatching.use_gpu", "1" if use_gpu else "0",
        ]
    elif matcher == "sequential":
        match_args = [
            "sequential_matcher",
            "--database_path", str(database_path),
            "--SiftMatching.use_gpu", "1" if use_gpu else "0",
        ]
    else:
        print(f"エラー: 未対応のマッチャー: {matcher}")
        sys.exit(1)
    run_colmap_command(match_args, log_file)
    print("Feature Matching 完了")

    # Step 3: Sparse Reconstruction (Mapper)
    print("\n=== Step 3: Sparse Reconstruction ===")
    mapper_args = [
        "mapper",
        "--database_path", str(database_path),
        "--image_path", str(image_dir),
        "--output_path", str(sparse_dir),
    ]
    run_colmap_command(mapper_args, log_file)
    print("Sparse Reconstruction 完了")

    # Step 4: Model Converter (バイナリ → テキスト)
    print("\n=== Step 4: Model Converter ===")
    sparse_model_dir = sparse_dir / "0"
    if not sparse_model_dir.exists():
        print(f"エラー: SfM モデルが生成されていません: {sparse_model_dir}")
        sys.exit(1)

    run_colmap_command([
        "model_converter",
        "--input_path", str(sparse_model_dir),
        "--output_path", str(sparse_model_dir),
        "--output_type", "TXT",
    ], log_file)
    print("Model Converter 完了")

    # 登録画像数の確認
    registered = count_registered_images(str(sparse_model_dir))
    ratio = registered / total_images * 100 if total_images > 0 else 0
    print(f"\n=== COLMAP 結果 ===")
    print(f"  登録画像数: {registered} / {total_images} ({ratio:.1f}%)")
    if ratio < 80:
        print("  警告: 登録率が 80% 未満です。撮影方法の見直しを検討してください。")
    else:
        print("  品質基準を満たしています。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="COLMAP 実行")
    parser.add_argument("--config", default="config.yaml", help="設定ファイルのパス")
    args = parser.parse_args()
    run_colmap(args.config)
