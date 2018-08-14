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
		--disable-summary \
		--disable-kb-edge \
		--input-layers 3 \
		--answer-classes 512 \
		--vocab-size 512 \
		--memory-transform-layers 1 \
		--max-decode-iterations 8 \
		--output-activation tanh \
		--output-layers 1 \
		--read-activation tanh \
		--read-layers 1 \
		--memory-forget-activation tanh \
		--control-dropout 0.0 \
		--read-dropout 0.0 \
		--input-width 64 \
		--learning-rate 0.001 \
		--max-gradient-norm 4 \
		--disable-dynamic-decode \
	"

# floyd run \
# 	--message "$COMMIT station adjacency - verify max accuracy" \
# 	--cpu \
# 	--env tensorflow-1.8 \
# 	--data davidmack/datasets/mac-graph-station-adjacent:/input \
# 	--max-runtime $RUNTIME \
# 	"python -m macgraph.train \
# 		--input-dir /input \
# 		--output-dir /output \
# 		--model-dir /output/model \
# 		--disable-summary \
# 		--disable-kb-node \
# 		--max-decode-iterations 1 \
# 		--input-layers 1 \
# 		--disable-memory-cell \
# 		--disable-control-cell \
# 		--disable-dynamic-decode \
# 		--disable-question-state \
# 		--read-activation abs \
# 		--read-from-question \
# 		--read-dropout 0.0 \
# 		--output-layers 1 \
# 		--answer-classes 8 \
# 		--input-width 64 \
# 		--output-activation mi \
# 		--learning-rate 0.001 \
# 		--max-gradient-norm 4.0 \
# 		--read-layers 1 \
# 		--read-indicator-rows 1 \
# 	"
