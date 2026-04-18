
~~~bash
python img/extract_img.py
python src/preprocess.py --config config.yaml
python src/run_colmap.py --config config.yaml
python src/prepare_training.py --config config.yaml
python src/train.py --config config.yaml
python src/render.py --config config.yaml
~~~

~~~bash
# COLMAP の既存データを削除して再実行
rm -rf img/frames
rm -rf data/processed
rm -rf data/colmap/
python src/run_colmap.py --config config.yaml

# 学習データ再準備
rm -rf data/training/
python src/prepare_training.py --config config.yaml

# 学習再実行
python src/train.py --config config.yaml
~~~
