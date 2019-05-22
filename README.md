# ocr_code
python:3.5
pytorch:0.4.1

数据准备：
create_dataset:
pip install lmdb
pip install opencv-python
python2
执行：
python create_labels.py
python create_dataset.py

执行训练：
python train.py
注意：在os.environ 以及gpu_list调整gpu数量

测试模型：
生成测试数据集以后，更改test.py的valroot路径
执行：python test.py
