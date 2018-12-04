#!/bin/bash

task=NearestStationArchitecture
tag=curriculum
iterations=4

python -m macgraph.regression_test \
		--name $task \
		--model-dir output/model/$task/$tag/$iterations \
		--tag $tag \
		--train-max-steps 50 \
		--max-decode-iterations $iterations \
		--enable-curriculum \
