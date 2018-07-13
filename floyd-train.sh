#!/bin/sh

floyd run \
	--gpu \
	--tensorboard \
	--env tensorflow-1.8 \
    --data davidmack/datasets/mac-graph:/input \
    --data davidmack/projects/mac-graph/19/output:/warm_start \
    "python -m mac-graph.train --input-dir /input --output-dir /output --model-dir /output/model --warm-start-dir /warm_start"