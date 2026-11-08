export PYTHONPATH := src
.PHONY: setup data train eda test serve all
setup: ; pip install -r requirements.txt
data:  ; python -m noshow.data_generator && python -m noshow.data_prep
train: ; python -m noshow.train
eda:   ; python -m noshow.eda
sample: ; python scripts/make_sample_bookings.py
test:  ; python -m pytest -q tests
serve: ; python -m uvicorn noshow.api:app --port 8000
all: data train eda sample test
