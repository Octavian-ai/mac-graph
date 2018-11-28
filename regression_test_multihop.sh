#!/bin/bash


tasks=(
	"HasCycle" 
	"NearestStationArchitecture" 
	"StationTwoHops" 
	"DistinctRoutes" 
	"StationShortestAvoidingCount" 
	"StationShortestCount" 
)

tag=sans_mp
iterations=1


for task in "${tasks[@]}"
do
	echo $task
	nohup python -m macgraph.regression_test \
		--name $task \
		--model-dir output/model/$task/$tag/$iterations \
		--tag $tag \
		--train-max-steps 50 \
		--max-decode-iterations $iterations \
		--disable-message-passing \
		--enable-comet  &
done
