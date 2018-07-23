#!/bin/sh

floyd run \
    --message "Baseline - early attn q dense" \
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
        --max-decode-iterations 2 \
        --max-steps 50000
    "
