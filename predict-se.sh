#!/bin/sh

COMMIT=$(git --no-pager log --pretty=format:'%h' -n 1)

python -m macgraph.predict \
	--model-dir output/model/se/$COMMIT \
	--input-dir input_data/processed/se_50k_small \
	--disable-dynamic-decode \
	--disable-question-state \
	--disable-memory-cell \
	$@