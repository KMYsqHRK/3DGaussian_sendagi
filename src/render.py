"""レンダリング・品質評価スクリプト

学習済み 3DGS モデルから任意視点の画像を生成し、PSNR/SSIM を計算する。
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Optional

import yaml


def render(config_path: str, model_path: Optional[str] = None) -> None:
    with open(config_path) as f:
        config = yaml.safe_load(f)

    root = Path(config["project"]["root"])
    training_dir = root / config["paths"]["training_data"]
    gs_dir = root / config["paths"]["gaussian_splatting"]
    render_output = root / config["paths"]["render_output"]
    render_cfg = config.get("render", {})
    experiment_name = config["project"]["name"]

    if model_path:
        model_dir = Path(model_path)
    else:
        model_dir = root / config["paths"]["model_output"] / experiment_name

    if not model_dir.exists():
        print(f"エラー: モデルが見つかりません: {model_dir}")
        sys.exit(1)

    gs_render_script = gs_dir / "render.py"
    gs_metrics_script = gs_dir / "metrics.py"

    if not gs_render_script.exists():
        print(f"エラー: 3DGS 公式実装が見つかりません: {gs_dir}")
        sys.exit(1)

    # レンダリング実行
    print("=== レンダリング開始 ===")
    print(f"モデル: {model_dir}")

    cmd = [
        sys.executable, str(gs_render_script),
        "-m", str(model_dir),
    ]

    if render_cfg.get("skip_train", False):
        cmd.append("--skip_train")
    if render_cfg.get("skip_test", False):
        cmd.append("--skip_test")

    print(f"コマンド: {' '.join(cmd)}\n")
    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"エラー: レンダリングが失敗しました (exit code: {result.returncode})")
        sys.exit(1)

    print("レンダリング完了")

    # 品質評価（metrics.py が存在する場合）
    if gs_metrics_script.exists():
        print("\n=== 品質評価 ===")
        metrics_cmd = [
            sys.executable, str(gs_metrics_script),
            "-m", str(model_dir),
        ]
        print(f"コマンド: {' '.join(metrics_cmd)}\n")
        metrics_result = subprocess.run(metrics_cmd)

        if metrics_result.returncode != 0:
            print("警告: 品質評価でエラーが発生しました。")
        else:
            print("品質評価完了")
    else:
        print("\n注意: metrics.py が見つかりません。品質評価をスキップします。")

    # レンダリング結果のコピー
    rendered_train = model_dir / "train"
    rendered_test = model_dir / "test"
    dest = render_output / experiment_name
    dest.mkdir(parents=True, exist_ok=True)

    for src_dir, name in [(rendered_train, "train"), (rendered_test, "test")]:
        if src_dir.exists():
            print(f"  レンダリング結果 ({name}): {src_dir}")

    print(f"\n=== 完了 ===")
    print(f"モデル: {model_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3DGS レンダリング・品質評価")
    parser.add_argument("--config", default="config.yaml", help="設定ファイルのパス")
    parser.add_argument("--model_path", default=None, help="モデルディレクトリのパス")
    args = parser.parse_args()
    render(args.config, args.model_path)
