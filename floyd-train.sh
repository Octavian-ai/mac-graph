#!/bin/bash

RUNTIME=$(expr 60 \* 60 \* 1)
COMMIT=$(git --no-pager log --pretty=format:'%h' -n 1)

for i in 1 2 3
do

floyd run \
	--message "$COMMIT baseline - abs [repeat $i]" \
	--cpu \
	--env tensorflow-1.8 \
	--data davidmack/datasets/mac-graph-station-adjacent:/input \
	--max-runtime $RUNTIME  \
	"python -m mac-graph.train \
		--input-dir /input \
		--output-dir /output \
		--model-dir /output/model \
		--disable-kb-node \
		--max-decode-iterations 1 \
		--input-layers 1 \
		--disable-memory-cell \
		--disable-control-cell \
		--disable-dynamic-decode \
		--disable-question-state \
		--disable-read-abs \
		--read-dropout 0.0 \
	"

done