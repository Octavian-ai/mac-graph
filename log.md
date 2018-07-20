# Experiment log

Notes on past experiments and results.


## Question level progress

### Station Properties

This is the first (easy) question I've got working. Here's a log of known working configurations:

- Commit `85b98a3`
	- `python -m mac-graph.train --input-dir input_data/processed/stationProp_tiny_50k_12th --model-dir output/model_sp_85b98a3_v512 --kb-edge-width 7 --disable-data-stack --disable-kb-edge --vocab-size 512 --answer-classes 512`
	-  LR 0.001, batch 32, embed_width 64, control_width 64, memory_width 64, decode_iterations 8, num_input_layers 3, vocab_size 512, max_gradient_norm 4.0, dropout 2.0
	- 90% accuracy after 10k training steps

- Commit `5eef6f0`
	- Added kb_edges and data_stack
	- `python -m mac-graph.train --input-dir input_data/processed/stationProp_tiny_50k_12th`
	- 80% acc after 6k steps


- Commit `da4b306`
	- Adding residual connections increased accuracy 25%, achieving 100% after 10k steps
	- `./train-station-properties.sh`


### Station Adjacency

This is a variation of station-properties, where we need to retrieve from the edge-list
whether a certain edge exists. 

The successful station property model does no better than random guessing. I'm exploring a range of extra operators to make this task possible.

- Commit `e820ae9`
	- Increasing `--memory-transform-layers` from 1 to many (e.g. 13) seems to help
	- 66% after 10k steps, then plateaus (earlier similar code seen achieving 70% after 3hrs)
	- ```python -m mac-graph.train \
		--input-dir input_data/processed/sa_small_100k_balanced \
		--model-dir output/model/sa/$COMMIT \
		--log-level DEBUG \
		--disable-kb-node \
		--disable-data-stack \
		--disable-indicator-row \
		--disable-read-comparison \
		--memory-transform-layers 13 \
		--memory-width 128```
