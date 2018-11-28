#!/bin/bash

task=StationShortestCount
tag=sans_read_edge
iterations=1

python -m macgraph.regression_test \
		--name $task \
		--model-dir output/model/$task/$tag/$iterations \
		--tag $tag \
		--train-max-steps 50 \
		--max-decode-iterations $iterations \
		--disable-kb-edge \
		--enable-comet