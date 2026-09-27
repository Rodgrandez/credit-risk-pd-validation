# Makefile
PY ?= python
.PHONY: data models report all test lint
data:   ; $(PY) -m credit_risk.pipeline data
models: ; $(PY) -m credit_risk.pipeline models
report: ; $(PY) -m credit_risk.pipeline report
all:    ; $(PY) -m credit_risk.pipeline all
test:   ; $(PY) -m pytest -q
lint:   ; ruff check src tests
