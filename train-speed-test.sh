
MAX=2000

for i in 4 5 6 
do
	python -m macgraph.train \
		--input-dir input_data/processed/sa_small_100k_balanced \
		--model-dir output/model/speed/dynamic/$i \
		--log-level DEBUG \
		--disable-kb-node \
		--max-steps $MAX

	python -m macgraph.train \
		--input-dir input_data/processed/sa_small_100k_balanced \
		--model-dir output/model/speed/static/$i \
		--log-level DEBUG \
		--disable-kb-node \
		--disable-dynamic-decode \
		--max-steps $MAX

done



