#!/bin/sh

RUNTIME=$(expr 60 \* 60 \* 1)


floyd run \
	--message "Baseline + 2 decode - embed 16" \
	--cpu \
	--tensorboard \
	--env tensorflow-1.8 \
	--data davidmack/datasets/mac-graph-station-adjacent:/input \
	--max-runtime $RUNTIME  \
	"python -m mac-graph.train \
		--input-dir /input \
		--output-dir /output \
		--model-dir /output/model \
		--embed-width 16 \
		--disable-kb-node --max-decode-iterations 2 --num-input-layers 1 --enable-indicator-row --disable-memory-cell
	"
