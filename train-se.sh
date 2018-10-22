#!/bin/sh

COMMIT=$(git --no-pager log --pretty=format:'%h' -n 1)

python -m macgraph.train \
	--model-dir output/model/se/$COMMIT \
	--input-dir input_data/processed/se_50k_small \
	--max-decode-iterations 1 \
	$@