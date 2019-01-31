#!/bin/bash

task=StationShortestAvoidingCount
iteration=4

python -m macgraph.train \
	--dataset $task \
	--tag iter_$iteration \
	--tag upto_6 \
	--tag s1 \
	--tag r$RANDOM \
	--filter-output-class 0 \
	--filter-output-class 1 \
	--filter-output-class 2 \
	--filter-output-class 3 \
	--filter-output-class 4 \
	--filter-output-class 5 \
	--filter-output-class 6 \
	--train-max-steps 5 \
	--max-decode-iterations $iteration \
	--disable-control-cell \
	--disable-read-cell \
	--disable-memory-cell \
	--disable-input-bilstm \
	--disable-mp-gru \
	--enable-embed-const-eye \
	--eval-every 500 \
	--mp-state-width 1 \
	--input-width 128 \
	--embed-width 128 \
	--learning-rate 0.1 \
	--fast \