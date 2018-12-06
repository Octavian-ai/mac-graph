#!/bin/bash


python -m macgraph.profile \
	--dataset StationShortestCount \
	--max-decode-iterations 2 \
	--train-max-steps 1 \
	--tag baseline_profile \
	--tag upto_4 \
	--tag iter_2 \
	--filter-output-class 0 \
	--filter-output-class 1 \
	--filter-output-class 2 \

