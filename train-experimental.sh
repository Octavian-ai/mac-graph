
#!/bin/sh

COMMIT=$(git --no-pager log --pretty=format:'%h' -n 1)

python -m mac-graph.train \
	--input-dir input_data/processed/sa_small_100k_balanced \
	--model-dir output/model/sa/$COMMIT \
	--log-level DEBUG \
	--disable-kb-node \
	--max-decode-iterations 1 \
	--num-input-layers 1 \
	--enable-indicator-row \
	--disable-memory-cell \

	

