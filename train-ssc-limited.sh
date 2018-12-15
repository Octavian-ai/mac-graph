#!/bin/bash

python -m macgraph.train \
	--dataset StationShortestCount \
	--max-decode-iterations 1 \
	--train-max-steps 6 \
	--tag upto_1 \
	--tag iter_1 \
	--tag mp_simple \
	--tag with_fixed_embed \
	--tag with_both_q \
	--filter-output-class 1 \
	--filter-output-class 0 \
	--control-heads 1 \
	--disable-control-cell \
	--disable-memory-cell \
	--disable-read-cell \
	--enable-embed-const-eye \
	--eval-every 90 \
	--mp-state-width 1 \
	--disable-mp-gru \
	--disable-input-bilstm \
	--enable-summary-image \
	--input-width 128 \
	--embed-width 128 \
	--learning-rate 1.0 \

