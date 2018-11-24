#!/bin/bash


tasks=(
	"CountCycles" 
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
	nohup python -m macgraph.regression_test --name $task --log-level=DEBUG --train-steps 50 &
done
