#!/bin/bash


tasks=(
	"HasCycle" 
	"NearestStationArchitecture" 
	"StationTwoHops" 
	"DistinctRoutes" 
	"StationShortestAvoidingCount" 
	"StationShortestCount" 
)

tag=curriculum

iterations=(
	"4"
)

for task in "${tasks[@]}"
do
	for iteration in "${iterations[@]}"
	do
		nohup python -m macgraph.regression_test \
			--name $task \
			--model-dir output/model/$task/$tag/$iteration \
			--tag $tag \
			--train-max-steps 50 \
			--max-decode-iterations $iteration \
			--enable-curriculum \
			--enable-comet &> nohup-$task-$iteration.out&
	done
done
