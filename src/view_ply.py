"""3DGS 点群ビューア

学習済み .ply ファイルを matplotlib で3D表示する。
約470万点はそのままだと重いため、ランダムサンプリングして表示。
マウス操作: 左ドラッグ=回転, スクロール=ズーム
"""

import argparse
import numpy as np
from pathlib import Path
from plyfile import PlyData


def load_gaussian_ply(ply_path: str):
    """3DGS の .ply から位置と色を読み込む。"""
    plydata = PlyData.read(ply_path)
    vertex = plydata["vertex"]

    # 位置
    xyz = np.stack([
        np.array(vertex["x"]),
        np.array(vertex["y"]),
        np.array(vertex["z"]),
    ], axis=1)

    # 色（SH の 0次係数から RGB に変換）
    if "f_dc_0" in vertex.data.dtype.names:
        C0 = 0.28209479177387814  # SH 0次の係数
        r = 0.5 + C0 * np.array(vertex["f_dc_0"])
        g = 0.5 + C0 * np.array(vertex["f_dc_1"])
        b = 0.5 + C0 * np.array(vertex["f_dc_2"])
        colors = np.stack([r, g, b], axis=1)
        colors = np.clip(colors, 0.0, 1.0)
    elif "red" in vertex.data.dtype.names:
        colors = np.stack([
            np.array(vertex["red"]),
            np.array(vertex["green"]),
            np.array(vertex["blue"]),
        ], axis=1) / 255.0
    else:
        colors = np.ones((len(xyz), 3)) * 0.5

    return xyz, colors


def view(ply_path: str, max_points: int = 100000):
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt

    print(f"読み込み中: {ply_path}")
    xyz, colors = load_gaussian_ply(ply_path)
    total = len(xyz)
    print(f"ガウシアン数: {total:,}")

    # サンプリング（全点表示は重すぎるため）
    if total > max_points:
        idx = np.random.choice(total, max_points, replace=False)
        xyz = xyz[idx]
        colors = colors[idx]
        print(f"表示点数: {max_points:,} ({max_points/total*100:.1f}% をランダムサンプリング)")
    else:
        print(f"表示点数: {total:,}")

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        xyz[:, 0], xyz[:, 1], xyz[:, 2],
        c=colors,
        s=0.1,
        marker=".",
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"3DGS Point Cloud ({max_points:,} / {total:,} points)")

    print("\n操作方法:")
    print("  左ドラッグ  = 回転")
    print("  スクロール  = ズーム")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3DGS 点群ビューア")
    parser.add_argument(
        "--ply",
        default="output/models/room_archive/point_cloud/iteration_30000/point_cloud.ply",
        help=".ply ファイルのパス",
    )
    parser.add_argument(
        "--points",
        type=int,
        default=100000,
        help="表示する最大点数 (デフォルト: 100,000)",
    )
    args = parser.parse_args()
    view(args.ply, args.points)
