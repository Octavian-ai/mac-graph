#!/bin/sh

COMMIT=$(git --no-pager log --pretty=format:'%h' -n 1)

python -m macgraph.train \
	--log-level DEBUG \
	--model-dir output/model/sp/$COMMIT \
	--input-dir input_data/processed/stationProp_tiny_50k_12th \
	--disable-kb-edge \
	--kb-edge-width 7