#!/bin/bash

task=StationShortestCount
tag=only_read
iterations=1

nohup python -m macgraph.regression_test \
		--name $task \
		--model-dir output/model/$task/$tag/$iterations \
		--tag $tag \
		--train-max-steps 50 \
		--max-decode-iterations $iterations \
		--disable-memory-cell \
		--disable-control-cell \
		--disable-message-passing \
		--enable-comet &

