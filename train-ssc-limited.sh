#!/bin/bash

python -m macgraph.train \
	--dataset StationShortestCount \
	--max-decode-iterations 2 \
	--train-max-steps 2 \
	--tag upto_1 \
	--tag iter_2 \
	--tag mp_simple \
	--tag with_fixed_embed \
	--filter-output-class 1 \
	--filter-output-class 0 \
	--disable-control-cell \
	--disable-read-cell \
	--enable-embed-const-eye \
	--eval-every 90 \
	--mp-state-width 4 \
	--disable-mp-gru \
	--mp-read-heads 4 \
	--memory-width 4 \
	--disable-memory-cell \
	--disable-input-bilstm \
	--input-width 128 \
	--embed-width 128 \
	--learning-rate 1.0 \
	--fast \

