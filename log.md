# Experiment log

Notes on past experiments and results.


## Question level progress

### Station Properties

This is the first (easy) question I've got working. Here's a log of known working configurations:

- Commit <kb>85b98a3</kb>
	- `python -m mac-graph.train --input-dir input_data/processed/stationProp_tiny_50k_12th --model-dir output/model_sp_85b98a3_v512 --kb-edge-width 7 --disable-data-stack --disable-kb-edge --vocab-size 512 --answer-classes 512`
	- 90% accuracy after 10k training steps

- Commit <kb>5eef6f0</kb>
	- Added kb_edges and data_stack
	- `python -m mac-graph.train --input-dir input_data/processed/stationProp_tiny_50k_12th`
	- 80% acc after 6k steps


### Station adjacency

- `be6bd07`
	- 64% accuracy after 14k training steps
	- Suspected will struggle to rapidly gain accuracy based on other runs of same code

- `da4b306`
	- Adding residual connections increased accuracy 25%, achieving 100% after 10k steps
	- `./train-cmp-station-prop.sh`

