#!/bin/bash

RUNTIME=$(expr 60 \* 60 \* 2)
COMMIT=$(git --no-pager log --pretty=format:'%h' -n 1)

floyd run \
	--message "$COMMIT experiment" \
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
		--num-input-layers 1 \
		--disable-memory-cell \
		--memory-width 8 \
	"
