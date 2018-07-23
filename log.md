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

- Commit `636d354` as above


### Station Adjacency

This is a variation of station-properties, where we need to retrieve from the edge-list
whether a certain edge exists. 

The successful station property model does no better than random guessing. I'm exploring a range of extra operators to make this task possible.

#### Thoughts
- The problem is the network cannot formulate the query (test this!)
- I believe that it's easy in embedding to detect none-exists as you can just set two bits on the records vs other records and amplify it using non-linearity. 
- You don't need extra indicator row since every bit of unused vocab is one such row. 
- Larger vocab = diluting down the attention >> maybe that is why it improves station properties

#### Proof points

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

- Commit `be6bd07` branch `feature-station-adj-known-ok`
	- 64% accuracy after 14k training steps
	- Suspected will struggle to rapidly gain accuracy based on other runs of same code

- Commit `92d3146`
	- Removing residual connections everywhere maxing out at 62% acc

- Commit `d4bb4c9`
	- Equal best evar

- Commit `eebb0e8`
	- Embed width 32 worked as well as 64. 128 failed.

- Commit `6e64305`
	- Switched back to residual_depth=2 and it worked ok
	- Decode iterations 4

- `6fdc835`: 
	- Was using static decode when I believed dynamic decode
	- Relu activation on deeep failed
	- Memory cell sigmoid for forget signal made no difference (was in class of best performance seen so far)

- `fcb13d0`: 
	- Was using static decode when I believed dynamic decode
	- Same performance, smaller network, faster convergence

- `e120afc`: **gold standard so far**
	- Was using static decode when I believed dynamic decode
	- Fastest training seen (72% after 42min). 
	- Achieved 80% after 600k (5hrs) training steps
	- Should achieve 100% in 10 min!

- `956b4f6`:
	- `--vocab-size 90 --answer-classes 8` seen to train alright
	- Fastest ever accuracy growth {'log_level': 'DEBUG', 'output_dir': './output', 'input_dir': 'input_data/processed/sa_small_100k_balanced', 'model_dir': 'output/model/sa/956b4f6', 'limit': None, 'type_string_prefix': None, 'eval_holdback': 0.1, 'predict_holdback': 0.005, 'warm_start_dir': None, 'batch_size': 32, 'max_steps': None, 'max_gradient_norm': 4.0, 'learning_rate': 0.001, 'dropout': 0.2, 'answer_classes': 8, 'vocab_size': 90, 'embed_width': 32, 'num_input_layers': 3, 'kb_node_width': 7, 'kb_edge_width': 3, 'read_heads': 1, 'data_stack_width': 64, 'data_stack_len': 20, 'control_width': 64, 'memory_width': 128, 'memory_transform_layers': 2, 'use_kb_node': False, 'use_kb_edge': True, 'use_data_stack': False, 'use_control_cell': True, 'use_dynamic_decode': True, 'max_decode_iterations': 2, 'modes': ['eval', 'train', 'predict'], 'eval_input_path': 'input_data/processed/sa_small_100k_balanced/eval_input.tfrecords', 'train_input_path': 'input_data/processed/sa_small_100k_balanced/train_input.tfrecords', 'predict_input_path': 'input_data/processed/sa_small_100k_balanced/predict_input.tfrecords', 'all_input_path': 'input_data/processed/sa_small_100k_balanced/all_input.tfrecords', 'vocab_path': 'input_data/processed/sa_small_100k_balanced/vocab.txt', 'question_types_path': 'input_data/processed/sa_small_100k_balanced/types.yaml', 'answer_classes_path': 'input_data/processed/sa_small_100k_balanced/answer_classes.yaml'}



## Notes on training infrastructure

- FloydHub seems to fail with static decoding, need to do dynamic
- Note: Due to bug in `6fdc835` I'm not certain of these results
- Steps per second:
	- FloydHub 
		- GPU
			- Dynamic 9.6, 19.6 observed
			- Static 11.9 - But fails to increase test accuracy
		- CPU
			- Dynamic 7.8
	- MacBook Pro 
		- Dynamic 51, 53, 52, 23, 21, 23
		- Static 30, 30, 30, 30, 29, 30

