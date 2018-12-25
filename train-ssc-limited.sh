#!/bin/bash

# for i in `seq 1 10`;
# do

i=1
python -m macgraph.train \
	--dataset StationShortestCount \
	--max-decode-iterations 2 \
	--train-max-steps 5 \
	--tag upto_1 \
	--tag iter_2 \
	--tag mp_r4 \
	--tag w_memory \
	--tag $i \
	--filter-output-class 1 \
	--filter-output-class 0 \
	--disable-control-cell \
	--disable-read-cell \
	--disable-input-bilstm \
	--enable-embed-const-eye \
	--eval-every 90 \
	--mp-state-width 1 \
	--disable-mp-gru \
	--mp-read-heads 2 \
	--memory-width 4 \
	--input-width 128 \
	--embed-width 128 \
	--learning-rate 0.1 \
	--random-seed $RANDOM \
	--fast
# done    



