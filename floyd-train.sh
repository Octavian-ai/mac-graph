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

# floyd run \
# 	--message "$COMMIT station properties" \
# 	--cpu \
# 	--env tensorflow-1.8 \
# 	--data davidmack/datasets/mac-graph-station-properties:/input \
# 	--max-runtime $RUNTIME \
# 	"python -m macgraph.train \
# 		--input-dir /input \
# 		--output-dir /output \
# 		--model-dir /output/model \
# 		--disable-kb-edge \
# 		--kb-edge-width 7 \
# 	"

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
		--disable-control-cell \
		--disable-dynamic-decode \
		--disable-question-state \
		--read-activation mi \
		--read-from-question \
		--read-dropout 0.0 \
		--output-layers 1 \
		--answer-classes 8 \
		--input-width 64 \
		--output-activation mi \
		--learning-rate 0.001 \
		--max-gradient-norm 4.0 \
		--read-layers 1
	"
