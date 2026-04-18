"""3DGS 学習実行スクリプト

3DGS 公式実装の train.py をラップして実行する。
設定ファイルからハイパーパラメータを読み込み、学習を制御する。
"""

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


def train(config_path: str) -> None:
    with open(config_path) as f:
        config = yaml.safe_load(f)

    root = Path(config["project"]["root"])
    training_dir = root / config["paths"]["training_data"]
    model_output = root / config["paths"]["model_output"]
    gs_dir = root / config["paths"]["gaussian_splatting"]
    experiment_name = config["project"]["name"]

    train_cfg = config["training"]
    output_path = model_output / experiment_name
    output_path.mkdir(parents=True, exist_ok=True)

    # 入力チェック
    if not (training_dir / "images").exists():
        print(f"エラー: 学習データが見つかりません: {training_dir}")
        print("先に prepare_training.py を実行してください。")
        sys.exit(1)

    gs_train_script = gs_dir / "train.py"
    if not gs_train_script.exists():
        print(f"エラー: 3DGS 公式実装が見つかりません: {gs_dir}")
        print("以下のコマンドでクローンしてください:")
        print(f"  git clone https://github.com/graphdeco-inria/gaussian-splatting --recursive {gs_dir}")
        sys.exit(1)

    # 3DGS train.py の引数を構築
    cmd = [
        sys.executable, str(gs_train_script),
        "-s", str(training_dir),
        "-m", str(output_path),
        "--iterations", str(train_cfg["iterations"]),
        "--densify_until_iter", str(train_cfg["densify_until_iter"]),
        "--densify_from_iter", str(train_cfg["densify_from_iter"]),
        "--densification_interval", str(train_cfg["densification_interval"]),
        "--opacity_reset_interval", str(train_cfg["opacity_reset_interval"]),
        "--position_lr_init", str(train_cfg["position_lr_init"]),
        "--position_lr_final", str(train_cfg["position_lr_final"]),
        "--feature_lr", str(train_cfg["feature_lr"]),
        "--sh_degree", str(train_cfg["sh_degree"]),
        "--resolution", str(train_cfg["resolution"]),
    ]

    # save_iterations
    save_iters = train_cfg.get("save_iterations", [])
    if save_iters:
        cmd += ["--save_iterations"] + [str(i) for i in save_iters]

    # test_iterations
    test_iters = train_cfg.get("test_iterations", [])
    if test_iters:
        cmd += ["--test_iterations"] + [str(i) for i in test_iters]

    # 背景色
    if train_cfg.get("white_background", False):
        cmd.append("--white_background")

    print("=== 3DGS 学習開始 ===")
    print(f"データ: {training_dir}")
    print(f"出力先: {output_path}")
    print(f"イテレーション: {train_cfg['iterations']}")
    print(f"SH 次数: {train_cfg['sh_degree']}")
    print(f"解像度スケール: 1/{train_cfg['resolution']}")
    print(f"\nコマンド: {' '.join(cmd)}\n")

    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"\nエラー: 3DGS 学習が失敗しました (exit code: {result.returncode})")
        sys.exit(1)

    # 成果物確認
    ply_path = output_path / "point_cloud" / f"iteration_{train_cfg['iterations']}" / "point_cloud.ply"
    if ply_path.exists():
        size_mb = ply_path.stat().st_size / (1024 * 1024)
        print(f"\n=== 学習完了 ===")
        print(f"  モデル: {ply_path}")
        print(f"  サイズ: {size_mb:.1f} MB")
    else:
        print(f"\n警告: 最終モデルが見つかりません: {ply_path}")
        print("中間保存されたモデルを確認してください。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3DGS 学習実行")
    parser.add_argument("--config", default="config.yaml", help="設定ファイルのパス")
    args = parser.parse_args()
    train(args.config)
