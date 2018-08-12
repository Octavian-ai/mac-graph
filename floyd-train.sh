#!/bin/bash

RUNTIME=$(expr 60 \* 60 \* 1)
COMMIT=$(git --no-pager log --pretty=format:'%h' -n 1)


# floyd run \
# 	--message "$COMMIT station properties + station adjacency" \
# 	--cpu \
# 	--env tensorflow-1.8 \
# 	--data davidmack/datasets/mac-graph-sa-sp:/input \
# 	--max-runtime $RUNTIME \
# 	"python -m macgraph.train \
# 		--input-dir /input \
# 		--output-dir /output \
# 		--model-dir /output/model \
# 	"

floyd run \
	--message "$COMMIT station properties" \
	--cpu \
	--env tensorflow-1.8 \
	--data davidmack/datasets/mac-graph-station-properties:/input \
	--max-runtime $RUNTIME \
	"python -m macgraph.train \
		--input-dir /input \
		--output-dir /output \
		--model-dir /output/model \
		--disable-kb-edge \
		--kb-edge-width 7 \
	"

floyd run \
	--message "$COMMIT station adjacency" \
	--cpu \
	--env tensorflow-1.8 \
	--data davidmack/datasets/mac-graph-station-adjacent:/input \
	--max-runtime $RUNTIME \
	"python -m macgraph.train \
		--input-dir /input \
		--output-dir /output \
		--model-dir /output/model \
		--disable-kb-node \
		--max-decode-iterations 1 \
		--input-layers 1 \
		--disable-memory-cell \
		--read-indicator-rows 1 \
		--disable-control-cell \
		--disable-dynamic-decode \
		--disable-question-state \
		--read-dropout 0.0 \
	"
