#!/bin/bash

RUNTIME=$(expr 60 \* 60 \* 2)
COMMIT=$(git --no-pager log --pretty=format:'%h' -n 1)


floyd run \
	--message "$COMMIT baseline without subtract dense(in_signal) prior to relu-abs" \
	--cpu \
	--env tensorflow-1.8 \
	--data davidmack/datasets/mac-graph-station-adjacent:/input \
	--max-runtime $(expr 60 \* 60 \* 1)  \
	"python -m mac-graph.train \
		--input-dir /input \
		--output-dir /output \
		--model-dir /output/model \
		--disable-kb-node --max-decode-iterations 1 --input-layers 1 --disable-memory-cell --read-indicator-rows 1 --disable-control-cell --disable-dynamic-decode --disable-question-state --read-dropout 0.0 \
	"


floyd run \
	--message "$COMMIT baseline + dropout, dynamic decode, question state, control cell" \
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
		--learning-rate 2E-6 \
		--max-gradient-norm 1.0 \
	"

floyd run \
	--message "$COMMIT baseline + dropout, dynamic decode, question state, control cell, memory cell" \
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
		--learning-rate 2E-6 \
		--max-gradient-norm 1.0 \
	"

floyd run \
	--message "$COMMIT baseline + dropout, dynamic decode, question state, control cell, memory cell, input-layers 3" \
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
		--input-layers 3 \
		--learning-rate 2E-6 \
		--max-gradient-norm 1.0 \
	"

floyd run \
	--message "$COMMIT baseline + dropout, dynamic decode, question state, control cell, memory cell, input-layers 3, decode iterations 4" \
	--cpu \
	--env tensorflow-1.8 \
	--data davidmack/datasets/mac-graph-station-adjacent:/input \
	--max-runtime $RUNTIME  \
	"python -m mac-graph.train \
		--input-dir /input \
		--output-dir /output \
		--model-dir /output/model \
		--disable-kb-node \
		--max-decode-iterations 4 \
		--input-layers 3 \
		--learning-rate 2E-6 \
		--max-gradient-norm 1.0 \
	"

