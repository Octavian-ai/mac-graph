

TASKS = StationShortestCount StationShortestAvoidingCount StationTwoHops NearestStationArchitecture DistinctRoutes CountCycles StationProperty StationExistence StationAdjecent StationPairAdjacent StationArchitectureAdjacent StationOneApart



for taks in $TASKS
do
	nohup python -m macgraph.evaluate --name $task &
done