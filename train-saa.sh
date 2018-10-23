#!/bin/sh

COMMIT=$(git --no-pager log --pretty=format:'%h' -n 1)

python -m macgraph.train \
	--model-dir output/model/saa/$COMMIT \
	--input-dir input_data/processed/saa \
	--max-decode-iterations 3 \
	$@