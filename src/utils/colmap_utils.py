"""COLMAP ユーティリティ"""

import subprocess
import shutil
from pathlib import Path


def check_colmap_installed() -> bool:
    """COLMAP がインストールされているか確認する。"""
    return shutil.which("colmap") is not None


def run_colmap_command(args: list[str], log_file: str | None = None) -> subprocess.CompletedProcess:
    """COLMAP コマンドを実行し、結果を返す。"""
    cmd = ["colmap"] + args
    print(f"実行: {' '.join(cmd)}")

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a") as f:
            f.write(f"=== Command: {' '.join(cmd)} ===\n")
            f.write(f"Return code: {result.returncode}\n")
            if result.stdout:
                f.write(f"STDOUT:\n{result.stdout}\n")
            if result.stderr:
                f.write(f"STDERR:\n{result.stderr}\n")
            f.write("\n")

    if result.returncode != 0:
        print(f"エラー: {result.stderr}")
        raise RuntimeError(f"COLMAP コマンド失敗: {' '.join(cmd)}\n{result.stderr}")

    return result


def count_registered_images(sparse_dir: str) -> int:
    """COLMAP の images.txt から登録された画像数を数える。"""
    images_txt = Path(sparse_dir) / "images.txt"
    if not images_txt.exists():
        return 0
    count = 0
    with open(images_txt) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                count += 1
    # images.txt は 2行ペア（カメラ情報 + 2D点情報）
    return count // 2
