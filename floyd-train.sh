#!/bin/bash

iterations=1
task=StationShortestCount
tag=only_control

RUNTIME=$(expr 60 \* 60 \* 4)
COMMIT=$(git --no-pager log --pretty=format:'%h' -n 1)

floyd run \
	--message "$COMMIT $task $tag" \
	--cpu \
	--env tensorflow-1.10 \
	--data davidmack/datasets/mac-graph-ssc:/input \
	--max-runtime $RUNTIME \
	"python -m macgraph.regression_test \
		--name $task \
		--input-dir /input \
		--model-dir /output \
		--tag $tag \
		--train-max-steps 50 \
		--max-decode-iterations $iterations \
		--disable-memory-cell \
		--disable-read-cell \
		--disable-message-passing \
		--enable-comet \
		--enable-floyd \
	"

