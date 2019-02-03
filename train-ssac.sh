#!/bin/bash


iteration=7

tasks=(
	"StationShortestAvoidingCount" 
	# "StationShortestCount" 
)

for task in "${tasks[@]}"
do
	python -m macgraph.train \
		--dataset $task \
		\
		--tag iter_$iteration \
		--max-decode-iterations $iteration \
		\
		--tag upto_6 \
		--filter-output-class 0 \
		--filter-output-class 1 \
		--filter-output-class 2 \
		--filter-output-class 3 \
		--filter-output-class 4 \
		--filter-output-class 5 \
		--filter-output-class 6 \
		\
		--tag exp_mp_cell \
		\
		--tag grad_clip \
		--enable-gradient-clipping \
		\
		--tag r$RANDOM \
		\
		--train-max-steps 5 \
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

done
