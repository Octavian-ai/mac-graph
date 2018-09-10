#!/bin/bash

RUNTIME=$(expr 60 \* 60 \* 1)
COMMIT=$(git --no-pager log --pretty=format:'%h' -n 1)

floyd run \
	--message "$COMMIT station existence - tanh" \
	--cpu \
	--env tensorflow-1.8 \
	--data davidmack/datasets/mac-graph-se:/input \
	--max-runtime $RUNTIME \
	"python -m macgraph.train \
		--input-dir /input \
		--output-dir /output \
		--model-dir /output/model \
		--disable-dynamic-decode \
		--disable-question-state \
		--disable-memory-cell \
		--read-activation tanh \
	"

floyd run \
	--message "$COMMIT station existence - relu" \
	--cpu \
	--env tensorflow-1.8 \
	--data davidmack/datasets/mac-graph-se:/input \
	--max-runtime $RUNTIME \
	"python -m macgraph.train \
		--input-dir /input \
		--output-dir /output \
		--model-dir /output/model \
		--disable-dynamic-decode \
		--disable-question-state \
		--disable-memory-cell \
		--read-activation relu \
	"

floyd run \
	--message "$COMMIT station existence - tanh_abs" \
	--cpu \
	--env tensorflow-1.8 \
	--data davidmack/datasets/mac-graph-se:/input \
	--max-runtime $RUNTIME \
	"python -m macgraph.train \
		--input-dir /input \
		--output-dir /output \
		--model-dir /output/model \
		--disable-dynamic-decode \
		--disable-question-state \
		--disable-memory-cell \
		--read-activation tanh_abs \
	"


