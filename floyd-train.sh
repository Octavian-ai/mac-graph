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
		--max-decode-iterations 1 \
		--input-layers 1 \
		--answer-classes 86 \
		--vocab-size 86 \
	"
