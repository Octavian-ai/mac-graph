#!/bin/sh

floyd run \
	--gpu \
	--tensorboard \
	--env tensorflow-1.8 \
    --data davidmack/datasets/mac-graph:/input \
    "python -m mac-graph.train --input-dir /input --output-dir /output --model-dir /output/model"