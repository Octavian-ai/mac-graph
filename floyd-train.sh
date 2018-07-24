#!/bin/sh

RUNTIME=$(expr 60 \* 60 \* 1)


floyd run \
	--message "Baseline + 2 decode - indicator row" \
	--cpu \
	--tensorboard \
	--env tensorflow-1.8 \
	--data davidmack/datasets/mac-graph-station-adjacent:/input \
	--max-runtime $RUNTIME  \
	"python -m mac-graph.train \
		--input-dir /input \
		--output-dir /output \
		--model-dir /output/model \
		--disable-kb-node \
		--max-decode-iterations 2 \
		--num-input-layers 1 \
		--disable-memory-cell
	"
