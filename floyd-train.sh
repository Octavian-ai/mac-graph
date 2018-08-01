#!/bin/bash

RUNTIME=$(expr 60 \* 60 \* 4)
COMMIT=$(git --no-pager log --pretty=format:'%h' -n 1)



floyd run \
	--message "$COMMIT station properties + station adjacency" \
	--gpu \
	--env tensorflow-1.8 \
	--data davidmack/datasets/mac-graph-sa-sp:/input \
	--data davidmack/projects/mac-graph/107/output:/warm-start \
	--max-runtime $RUNTIME \
	"python -m mac-graph.train \
		--input-dir /input \
		--output-dir /output \
		--model-dir /output/model \
		--warm-start-dir /warm-start/model \
	"
