"""画像前処理スクリプト

入力画像を COLMAP および 3DGS 学習に適した形式に整える。
- 解像度統一（長辺を max_long_side 以下にリサイズ）
- 形式統一（JPEG）
- 連番リネーム（frame_XXXX.jpg）
- 低品質フレームの除去（ラプラシアン分散によるブレ検出）
"""

import argparse
import sys
from pathlib import Path

import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils.image_utils import compute_blur_score, resize_image, get_image_files


def preprocess(config_path: str) -> None:
    with open(config_path) as f:
        config = yaml.safe_load(f)

    root = Path(config["paths"].get("root", config["project"]["root"]))
    input_dir = root / config["paths"]["input_frames"]
    output_dir = root / config["paths"]["processed"]
    output_dir.mkdir(parents=True, exist_ok=True)

    pre_cfg = config["preprocess"]
    max_long_side = pre_cfg["max_long_side"]
    blur_threshold = pre_cfg["blur_threshold"]
    output_format = pre_cfg["output_format"]
    jpeg_quality = pre_cfg.get("jpeg_quality", 95)

    image_files = get_image_files(str(input_dir))
    if not image_files:
        print(f"エラー: 入力画像が見つかりません: {input_dir}")
        sys.exit(1)

    print(f"入力画像数: {len(image_files)}")
    print(f"リサイズ上限: {max_long_side}px")
    print(f"ブレ閾値: {blur_threshold}")

    kept = 0
    removed = 0

    for i, img_path in enumerate(tqdm(image_files, desc="前処理")):
        # ブレ検出
        blur_score = compute_blur_score(str(img_path))
        if blur_score < blur_threshold:
            removed += 1
            continue

        # リサイズ
        img = resize_image(str(img_path), max_long_side)

        # 保存
        kept += 1
        out_name = f"frame_{kept:04d}.{output_format}"
        out_path = output_dir / out_name

        if output_format.lower() in ("jpg", "jpeg"):
            img = img.convert("RGB")
            img.save(str(out_path), "JPEG", quality=jpeg_quality)
        else:
            img.save(str(out_path))

    print(f"\n前処理完了:")
    print(f"  保持: {kept} 枚")
    print(f"  除去: {removed} 枚（ブレ検出）")
    print(f"  出力先: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="画像前処理")
    parser.add_argument("--config", default="config.yaml", help="設定ファイルのパス")
    args = parser.parse_args()
    preprocess(args.config)
