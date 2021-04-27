# ocr_code
python:3.5
pytorch:0.4.1

data prepare：
create_dataset:
pip install lmdb
pip install opencv-python
python2
run for prepare data：
python create_labels.py
python create_dataset.py

training：
python train.py

test model：
python test.py

competition url：https://bbs.pinggu.org/thread-6734039-1-1.html
We won the third place, and this is our method to solve the paoblem.
