#!/bin/sh

floyd run \
    --message Speedtest: Dynamic decode \
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
        --disable-data-stack \
        --disable-read-comparison \
        --memory-transform-layers 13 \
        --memory-width 128 \
        --embed-width 32 \
        --max-steps 20000 \
        --dynamic-decode
    "
