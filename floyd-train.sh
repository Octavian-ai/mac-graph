#!/bin/bash

RUNTIME=$(expr 60 \* 60 \* 2)
COMMIT=$(git --no-pager log --pretty=format:'%h' -n 1)


floyd run \
	--message "$COMMIT baseline + dropout, dynamic decode, question state" \
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
	"

