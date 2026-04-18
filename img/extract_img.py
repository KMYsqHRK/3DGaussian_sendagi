import cv2
import os
import sys

def extract_frames(video_path: str, fps: float = 2.0, output_dir: str = "frames"):
    """
    動画から指定したFPSで画像を抽出する

    Args:
        video_path: 動画ファイルのパス
        fps: 抽出するフレームレート（デフォルト: 2.0）
        output_dir: 出力ディレクトリ名
    """
    # 動画ファイルの存在確認
    if not os.path.exists(video_path):
        print(f"エラー: '{video_path}' が見つかりません。")
        sys.exit(1)

    # 出力ディレクトリを作成（スクリプトと同じ階層）
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, output_dir)
    os.makedirs(output_path, exist_ok=True)

    # 動画を開く
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"エラー: 動画ファイルを開けませんでした: {video_path}")
        sys.exit(1)

    # 動画情報を取得
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / video_fps if video_fps > 0 else 0

    print(f"動画情報:")
    print(f"  ファイル    : {video_path}")
    print(f"  FPS         : {video_fps:.2f}")
    print(f"  総フレーム数: {total_frames}")
    print(f"  再生時間    : {duration:.2f} 秒")
    print(f"  抽出FPS     : {fps}")
    print(f"  出力先      : {output_path}")
    print()

    # 何フレームおきに抽出するか計算
    frame_interval = video_fps / fps
    saved_count = 0
    frame_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 抽出タイミングか判定
        if frame_index >= saved_count * frame_interval:
            filename = os.path.join(output_path, f"frame_{saved_count:05d}.jpg")
            cv2.imwrite(filename, frame)
            saved_count += 1

            # 進捗表示
            progress = (frame_index / total_frames * 100) if total_frames > 0 else 0
            print(f"\r  抽出中... {saved_count} 枚 ({progress:.1f}%)", end="", flush=True)

        frame_index += 1

    cap.release()
    print(f"\n\n完了！ {saved_count} 枚の画像を '{output_path}' に保存しました。")


if __name__ == "__main__":
    # スクリプトと同じ階層の Sendagi.mp4 を対象にする
    script_dir = os.path.dirname(os.path.abspath(__file__))
    video_file = os.path.join(script_dir, "Sendagi.mp4")

    extract_frames(
        video_path=video_file,
        fps=2.0,
        output_dir="frames"
    )