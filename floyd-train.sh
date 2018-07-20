#!/bin/sh

floyd run \
	--gpu \
	--tensorboard \
	--env tensorflow-1.8 \
    --data davidmack/datasets/mac-graph-station-adjacent:/input \
    --max-runtime $(expr 60 \* 60 \* 1) \
    "python -m mac-graph.train \
    	--input-dir /input \
    	--output-dir /output \
    	--model-dir /output/model \
        --disable-kb-node \
        --memory-transform-layers 14 \
        --memory-width 128
    "
