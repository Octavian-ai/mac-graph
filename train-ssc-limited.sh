#!/bin/bash

python -m macgraph.train \
	--dataset StationShortestCount \
	--max-decode-iterations 1 \
	--train-max-steps 10 \
	--tag upto_inf \
	--tag iter_1 \
	--tag kb_edges \
	--control-heads 2 \
	--disable-memory-cell \
	--eval-every 60 \
	--mp-state-width 1 \
	--disable-mp-gru \
	--enable-summary-image \
	--disable-message-passing \
	--disable-kb-node \
	--learning-rate 0.015 \



	# --filter-output-class 1 \
	# --filter-output-class 0 \
