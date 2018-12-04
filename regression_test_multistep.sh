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
		# nohup python -m macgraph.regression_test \
		# 	--name $task \
		# 	--model-dir output/model/$task/$tag/$iteration \
		# 	--tag $tag \
		# 	--train-max-steps 50 \
		# 	--max-decode-iterations $iteration \
		# 	--enable-curriculum \
		# 	--enable-comet &> nohup-$task-$iteration.out&

		nohup python -m macgraph.regression_test \
			--name $task \
			--model-dir output/model/$task/curriculum_baseline/$iteration \
			--tag curriculum_baseline \
			--train-max-steps 50 \
			--max-decode-iterations $iteration \
			--filter-output-class 0 \
			--filter-output-class 1 \
			--filter-output-class 2 \
			--filter-output-class 3 \
			--filter-output-class 4 \
			--enable-comet &> nohup-$task-$iteration.out&
	done
done
