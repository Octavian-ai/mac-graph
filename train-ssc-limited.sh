#!/bin/bash

python -m macgraph.train \
	--dataset StationShortestCount \
	--max-decode-iterations 1 \
	--train-max-steps 10 \
	--tag upto_1 \
	--tag iter_1 \
	--tag mp_vanilla \
	--filter-output-class 1 \
	--filter-output-class 0 \
	--control-heads 2 \
	--disable-memory-cell \
	--disable-read-cell \
	--eval-every 60 \
	--mp-state-width 1 \
	--disable-mp-gru \
	--enable-summary-image \
	--learning-rate 0.015 \

