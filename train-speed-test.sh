
MAX=10000

for i in 1 2 3 
do
	python -m mac-graph.train \
		--input-dir input_data/processed/sa_small_100k_balanced \
		--model-dir output/model/speed/static/$i \
		--log-level DEBUG \
		--disable-kb-node \
		--disable-data-stack \
		--disable-read-comparison \
		--memory-transform-layers 13 \
		--read-heads 1 \
		--memory-width 128 \
		--embed-width 32 \
		--max-steps $MAX

	python -m mac-graph.train \
		--input-dir input_data/processed/sa_small_100k_balanced \
		--model-dir output/model/speed/dynamic/$i \
		--log-level DEBUG \
		--disable-kb-node \
		--disable-data-stack \
		--disable-read-comparison \
		--memory-transform-layers 13 \
		--read-heads 1 \
		--memory-width 128 \
		--embed-width 32 \
		--max-steps $MAX \
		--dynamic-decode
done



