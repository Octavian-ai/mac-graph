#!/bin/bash


tasks=(
	"HasCycle" 
	"NearestStationArchitecture" 
	"StationTwoHops" 
	"DistinctRoutes" 
	"StationShortestAvoidingCount" 
	"StationShortestCount" 
)

tag=vanilla
iterations=(
	"1"
	"2"
	"4"
	"8"
)


for task in "${tasks[@]}"
do
	for iterations in "${iterations[@]}"
	do
		nohup python -m macgraph.regression_test \
			--name $task \
			--model-dir output/model/$task/$tag/$iterations \
			--tag $tag \
			--train-max-steps 50 \
			--max-decode-iterations $iterations \
			--enable-comet  &
	done
done
