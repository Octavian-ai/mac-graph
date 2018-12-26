#!/bin/bash


tasks=(
	"HasCycle" 
	"NearestStationArchitecture" 
	"StationTwoHops" 
	"DistinctRoutes" 
	"StationShortestAvoidingCount" 
	"StationShortestCount" 
)

iterations=(
	"20"
)

for task in "${tasks[@]}"
do
	for iteration in "${iterations[@]}"
	do

		nohup python -m macgraph.train \
			--dataset $task \
			--tag basic_gru \
			--train-max-steps 20 \
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
			--enable-comet &> nohup-$task-$iteration.out&
	done
done
