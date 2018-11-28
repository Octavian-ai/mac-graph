#!/bin/bash


tasks=(
	"HasCycle" 
	"NearestStationArchitecture" 
	"StationTwoHops" 
	"DistinctRoutes" 
	"StationShortestAvoidingCount" 
	"StationShortestCount" 
)


for task in "${tasks[@]}"
do
	echo $task
	nohup python -m macgraph.regression_test \
		--name $task \
		--tag dumb \
		--disable-memory-cell \
		--disable-control-cell \
		--disable-read-cell \
		--disable-message-passing \
		--log-level=DEBUG \
		--train-steps 50 \
		--max-decode-iterations 1 \
		--enable-comet  &
done
