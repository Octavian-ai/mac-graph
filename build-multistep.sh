#!/bin/bash

# python -m macgraph.input.build \
# 	--only-build-vocab \
# 	--gqa-paths input_data/raw/multistep/* \
# 	--name multistep && \

for j in StationShortestCount StationShortestAvoidingCount StationTwoHops NearestStationArchitecture DistinctRoutes CountCycles
	do
		nohup python -m macgraph.input.build \
			--skip-vocab \
			--vocab-path input_data/processed/vocab.txt \
			--gqa-paths input_data/raw/multistep/gqa-$j* \
			--name $j &
	done
