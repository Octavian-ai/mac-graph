#!/bin/bash

task=StationShortestCount
tag=sans_memory
iterations=1

nohup python -m macgraph.regression_test \
		--name $task \
		--model-dir output/model/$task/$tag/$iterations \
		--tag $tag \
		--train-max-steps 50 \
		--max-decode-iterations $iterations \
		--disable-memory-cell \
		--enable-comet &


tag=sans_control

nohup python -m macgraph.regression_test \
		--name $task \
		--model-dir output/model/$task/$tag/$iterations \
		--tag $tag \
		--train-max-steps 50 \
		--max-decode-iterations $iterations \
		--disable-control-cell \
		--enable-comet &