#!/bin/bash

python -m macgraph.train \
	--dataset StationShortestCount \
	--max-decode-iterations 1 \
	--train-max-steps 1 \
	--tag upto_2 \
	--tag iter_1 \
	--tag mp_nid \
	--filter-output-class 1 \
	--filter-output-class 2 \
	--disable-read-cell \
	--control-heads 2 \
	--disable-memory \
	--enable-mp-node-id \


python -m macgraph.train \
	--dataset StationShortestCount \
	--max-decode-iterations 1 \
	--train-max-steps 1 \
	--tag upto_2 \
	--tag iter_1 \
	--tag mp_rs \
	--filter-output-class 1 \
	--filter-output-class 2 \
	--disable-read-cell \
	--control-heads 2 \
	--disable-memory \
	--enable-mp-right-shift \


python -m macgraph.train \
	--dataset StationShortestCount \
	--max-decode-iterations 1 \
	--train-max-steps 1 \
	--tag upto_2 \
	--tag iter_1 \
	--tag mp_ngru \
	--filter-output-class 1 \
	--filter-output-class 2 \
	--disable-read-cell \
	--control-heads 2 \
	--disable-memory \
	--disable-mp-gru \


python -m macgraph.train \
	--dataset StationShortestCount \
	--max-decode-iterations 1 \
	--train-max-steps 1 \
	--tag upto_2 \
	--tag iter_1 \
	--tag mp_wide \
	--filter-output-class 1 \
	--filter-output-class 2 \
	--disable-read-cell \
	--control-heads 2 \
	--disable-memory \
	--mp-state-width 128 \



