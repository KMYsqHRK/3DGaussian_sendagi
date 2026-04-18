"""画像処理ユーティリティ"""

import cv2
import numpy as np
from pathlib import Path
from PIL import Image


def compute_blur_score(image_path: str) -> float:
    """ラプラシアン分散によるブレスコアを計算する。値が小さいほどブレが大きい。"""
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"画像を読み込めません: {image_path}")
    return cv2.Laplacian(img, cv2.CV_64F).var()


def resize_image(image_path: str, max_long_side: int) -> Image.Image:
    """長辺が max_long_side 以下になるようリサイズする。"""
    img = Image.open(image_path)
    w, h = img.size
    long_side = max(w, h)
    if long_side <= max_long_side:
        return img
    scale = max_long_side / long_side
    new_w = int(w * scale)
    new_h = int(h * scale)
    return img.resize((new_w, new_h), Image.LANCZOS)


def get_image_files(directory: str, extensions: tuple = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")) -> list[Path]:
    """指定ディレクトリから画像ファイルを取得しソートして返す。"""
    dir_path = Path(directory)
    files = []
    for ext in extensions:
        files.extend(dir_path.glob(f"*{ext}"))
        files.extend(dir_path.glob(f"*{ext.upper()}"))
    return sorted(set(files))
