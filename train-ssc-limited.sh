#!/bin/bash

# for i in `seq 1 10`;
# do

i=1
python -m macgraph.train \
	--dataset StationShortestCount \
	--max-decode-iterations 2 \
	--train-max-steps 2 \
	--tag upto_2 \
	--tag iter_2 \
	--tag gru_exp \
	--filter-output-class 0 \
	--filter-output-class 1 \
	--disable-control-cell \
	--disable-read-cell \
	--disable-memory-cell \
	--disable-input-bilstm \
	--enable-embed-const-eye \
	--eval-every 500 \
	--mp-state-width 1 \
	--input-width 128 \
	--embed-width 128 \
	--learning-rate 0.1 \
	--random-seed $RANDOM \
	--fast
# done    



