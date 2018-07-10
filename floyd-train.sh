#!/bin/sh

floyd run --gpu --tensorboard \
	--data davidmack/datasets/mac-graph:/input \
	--env tensorflow-1.8 \
	"python -m mac-graph.train --max-steps 4000 --input-dir /input --output-dir /output --model-dir /output/model"