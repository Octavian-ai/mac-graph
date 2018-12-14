#!/bin/bash

python -m macgraph.train \
	--dataset StationShortestCount \
	--max-decode-iterations 1 \
	--train-max-steps 20 \
	--tag upto_1 \
	--tag iter_1 \
	--tag mp_simple \
	--tag with_l2 \
	--filter-output-class 1 \
	--filter-output-class 0 \
	--control-heads 1 \
	--disable-memory-cell \
	--disable-read-cell \
	--eval-every 90 \
	--mp-state-width 1 \
	--disable-mp-gru \
	--disable-input-bilstm \
	--enable-summary-image \
	--input-width 64 \
	--embed-width 64 \
	--learning-rate 1.0 \

