#!/bin/sh

COMMIT=$(git --no-pager log --pretty=format:'%h' -n 1)

SIZE=100k

python -m macgraph.train \
	--model-dir output/model/s1a/$SIZE/$COMMIT \
	--input-dir input_data/processed/s1a_small_$SIZE \
	--max-decode-iterations 6 \
	--control-heads 2 \
	--read-heads 1 \
	$@