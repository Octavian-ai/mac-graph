#!/bin/bash


tasks=(
	"HasCycle" 
	"NearestStationArchitecture" 
	"StationTwoHops" 
	"DistinctRoutes" 
	"StationShortestAvoidingCount" 
	"StationShortestCount" 
	# "StationProperty" 
	# "StationExistence" 
	# "StationAdjacent" 
	# "StationPairAdjacent" 
	# "StationArchitectureAdjacent" 
	# "StationOneApart"
)


for task in "${tasks[@]}"
do
	echo $task
	nohup python -m macgraph.regression_test --name $task --log-level=DEBUG --train-steps 50 --max-decode-iterations 16 &
done
