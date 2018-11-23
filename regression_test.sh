#!/bin/bash


tasks=(
	"StationShortestCount" 
	"StationShortestAvoidingCount" 
	"StationTwoHops" 
	"NearestStationArchitecture" 
	"DistinctRoutes" 
	"CountCycles" 
	"StationProperty" 
	"StationExistence" 
	"StationAdjacent" 
	"StationPairAdjacent" 
	"StationArchitectureAdjacent" 
	"StationOneApart"
)


for task in "${tasks[@]}"
do
	echo $task
	nohup python -m macgraph.regression_test --name $task --log-level=WARN &
done
