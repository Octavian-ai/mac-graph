#!/bin/bash

task=StationShortestCount
tag=just_question_state
iterations=1

python -m macgraph.regression_test \
		--name $task \
		--model-dir output/model/$task/$tag/$iterations \
		--tag $tag \
		--train-max-steps 50 \
		--max-decode-iterations $iterations \
		--enable-question-state \
		--disable-memory-cell \
		--disable-read-cell \
		--disable-control-cell \
		--disable-message-passing \
		--enable-comet \
