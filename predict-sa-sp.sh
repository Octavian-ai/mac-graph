#!/bin/sh

COMMIT=$(git --no-pager log --pretty=format:'%h' -n 1)

python -m macgraph.predict \
	--model-dir output/model/sa_sp/$COMMIT \
	$@