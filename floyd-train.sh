#!/bin/bash

RUNTIME=$(expr 60 \* 60 \* 1)
COMMIT=$(git --no-pager log --pretty=format:'%h' -n 1)


floyd run \
	--message "$COMMIT station properties + station adjacency" \
	--cpu \
	--env tensorflow-1.8 \
	--data davidmack/datasets/mac-graph-sa-sp:/input \
	--max-runtime $RUNTIME \
	"python -m mac-graph.train \
		--input-dir /input \
		--output-dir /output \
		--model-dir /output/model \
	"

floyd run \
	--message "$COMMIT station properties" \
	--cpu \
	--env tensorflow-1.8 \
	--data davidmack/datasets/mac-graph-station-properties:/input \
	--max-runtime $RUNTIME \
	"python -m mac-graph.train \
		--input-dir /input \
		--output-dir /output \
		--model-dir /output/model \
		--kb-edge-width 7 \
	"

floyd run \
	--message "$COMMIT station adjacency" \
	--cpu \
	--env tensorflow-1.8 \
	--data davidmack/datasets/mac-graph-station-adjacent:/input \
	--max-runtime $RUNTIME \
	"python -m mac-graph.train \
		--input-dir /input \
		--output-dir /output \
		--model-dir /output/model \
	"
