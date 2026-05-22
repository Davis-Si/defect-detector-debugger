PYTHON := .venv/bin/python

.PHONY: setup data train analyze gradcam deployment benchmark all clean

setup:
	python3 -m venv --without-pip .venv
	.venv/bin/python /tmp/get-pip.py 2>/dev/null || curl -sSL https://bootstrap.pypa.io/get-pip.py | .venv/bin/python
	$(PYTHON) -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.1 torchvision==0.20.1
	$(PYTHON) -m pip install -r requirements.txt

data:
	mkdir -p data
	curl -sL -o data/train.parquet https://huggingface.co/datasets/newguyme/neu_cls/resolve/main/data/train-00000-of-00001.parquet
	curl -sL -o data/test.parquet https://huggingface.co/datasets/newguyme/neu_cls/resolve/main/data/test-00000-of-00001.parquet

train:
	$(PYTHON) -m src.train --augment none             --epochs 6 --run-name baseline
	$(PYTHON) -m src.train --augment flip             --epochs 6 --run-name flip
	$(PYTHON) -m src.train --augment flip_rotate      --epochs 6 --run-name flip_rotate
	$(PYTHON) -m src.train --augment flip_rotate_mild --epochs 6 --run-name flip_rotate_mild
	$(PYTHON) -m src.train --augment class_aware      --epochs 6 --run-name class_aware

analyze:
	$(PYTHON) -m src.analyze --runs runs/baseline runs/flip runs/flip_rotate runs/flip_rotate_mild runs/class_aware

gradcam:
	$(PYTHON) -m src.gradcam --run runs/baseline --mode all
	$(PYTHON) -m src.gradcam --run runs/class_aware --mode all

deployment:
	$(PYTHON) -m src.deployment --run runs/baseline

benchmark:
	$(PYTHON) -m src.benchmark --run runs/baseline

all: data train analyze gradcam deployment benchmark

clean:
	rm -rf runs/ reports/
