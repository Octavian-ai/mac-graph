#!/bin/bash


RUNTIME=$(expr 60 \* 60 \* 4)
COMMIT=$(git --no-pager log --pretty=format:'%h' -n 1)

floyd run \
	--message "$COMMIT ssc mp variations" \
	--cpu \
	--env tensorflow-1.10 \
	--data davidmack/datasets/mac-graph-ssc:/input \
	"python -m macgraph.train \
		--dataset StationShortestCount \
		--input-dir /input \
		--model-dir /output \
		--max-decode-iterations 1 \
		--train-max-steps 1 \
		--tag upto_2 \
		--tag iter_1 \
		--tag mp_nid \
		--filter-output-class 1 \
		--filter-output-class 2 \
		--disable-read-cell \
		--control-heads 2 \
		--disable-memory \
		--enable-mp-node-id \
		--enable-comet \
		--enable-floyd"


floyd run \
	--message "$COMMIT ssc mp variations" \
	--cpu \
	--env tensorflow-1.10 \
	--data davidmack/datasets/mac-graph-ssc:/input \
	"python -m macgraph.train \
		--dataset StationShortestCount \
		--input-dir /input \
		--model-dir /output \
		--max-decode-iterations 1 \
		--train-max-steps 1 \
		--tag upto_2 \
		--tag iter_1 \
		--tag mp_rs \
		--filter-output-class 1 \
		--filter-output-class 2 \
		--disable-read-cell \
		--control-heads 2 \
		--disable-memory \
		--enable-mp-right-shift \
		--enable-comet \
		--enable-floyd"


floyd run \
	--message "$COMMIT ssc mp variations" \
	--cpu \
	--env tensorflow-1.10 \
	--data davidmack/datasets/mac-graph-ssc:/input \
	"python -m macgraph.train \
		--dataset StationShortestCount \
		--input-dir /input \
		--model-dir /output \
		--max-decode-iterations 1 \
		--train-max-steps 1 \
		--tag upto_2 \
		--tag iter_1 \
		--tag mp_ngru \
		--filter-output-class 1 \
		--filter-output-class 2 \
		--disable-read-cell \
		--control-heads 2 \
		--disable-memory \
		--disable-mp-gru \
		--enable-comet \
		--enable-floyd"


floyd run \
	--message "$COMMIT ssc mp variations" \
	--cpu \
	--env tensorflow-1.10 \
	--data davidmack/datasets/mac-graph-ssc:/input \
	"python -m macgraph.train \
		--dataset StationShortestCount \
		--input-dir /input \
		--model-dir /output \
		--max-decode-iterations 1 \
		--train-max-steps 1 \
		--tag upto_2 \
		--tag iter_1 \
		--tag mp_wide \
		--filter-output-class 1 \
		--filter-output-class 2 \
		--disable-read-cell \
		--control-heads 2 \
		--disable-memory \
		--mp-state-width 128 \
		--enable-comet \
		--enable-floyd"
