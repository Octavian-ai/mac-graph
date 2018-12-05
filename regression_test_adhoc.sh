#!/bin/bash

task=StationShortestCount
tag=vanilla
iterations=2

python -m macgraph.regression_test \
		--dataset $task \
		--model-dir output/model/$task/$tag/$iterations \
		--tag $tag \
		--train-max-steps 50 \
		--max-decode-iterations $iterations \
