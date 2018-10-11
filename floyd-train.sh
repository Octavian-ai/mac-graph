#!/bin/bash

RUNTIME=$(expr 60 \* 60 \* 1)
COMMIT=$(git --no-pager log --pretty=format:'%h' -n 1)

floyd run \
	--message "$COMMIT s3a" \
	--gpu \
	--env tensorflow-1.10 \
	--data davidmack/datasets/mac-graph-ssc:/input \
	--max-runtime $RUNTIME \
	"python -m macgraph.train \
		--input-dir /input \
		--output-dir /output \
		--model-dir /output/model \
		\
		--filter-output-class 0 \
		--filter-output-class 1 \
		--filter-output-class 2 \
		--filter-output-class 3 \
		\
		--disable-dynamic-decode \
		--max-decode-iterations 8 \
		--control-heads 2 \
		--disable-input-bilstm \
		--input-width 64 \
		--embed-width 64 \
	"

