#!/bin/bash
python -m macgraph.input.build --input-dir input_data/processed/default --gqa-path input_data/raw/gqa-sa-sp-small-100k.yaml --limit 1000000 --skip-vocab
