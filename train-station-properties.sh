#!/bin/sh

python -m mac-graph.train --input-dir input_data/processed/stationProp_tiny_50k_12th --disable-data-stack --log-level DEBUG --model-dir output/model_sp --kb-edge-width 7 --disable-kb-edge