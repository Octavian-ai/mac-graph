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

- `da4b306`
	- Adding residual connections increased accuracy 25%, achieving 100% after 10k steps
	- `./train-cmp-station-prop.sh`

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

- `e120afc`:
	- Was using static decode when I believed dynamic decode
	- Fastest training seen (72% after 42min).
	- Achieved 80% after 600k (5hrs) training steps
	- Should achieve 100% in 10 min!

- `956b4f6`:
	- `--vocab-size 90 --answer-classes 8` seen to train alright
	- Fastest ever accuracy growth 88% after 3hr46 = 500k steps

- `8e4cc68`: **best so far**
	- saw better performance, but accidentally kept editing code. This is best so far

- `8312228`
	- 90% observed on FloydHub, cpu, one hour, 116k steps 
	- `--disable-kb-node --max-decode-iterations 2 --num-input-layers 1 --enable-indicator-row --disable-memory-cell`

- `7b810f1`: Same arch as above, 74% after 50k steps `./train-station-adj.sh`

- `4667b13`: 
	- Dropout improved things! best ever.
	- 91% after 200k steps

- `6253b97`:
	- Simplifed down to just a read and transform 
	- Highest ever 93% after 200k

- `a9651df`: embedding width 64 gave 97% accuracy
- `55f0039`: 
	- After 1 1/2 hours saw 99% max accuracy
	- Sometimes get NaN loss - LR reduction, max norm reduction, seem to help

- `d503000`: 
	- Achieved best ever performance on station adjacency task (97% 1hr on Floyd CPU, 99% after 1.5hrs on MBP)
	- This network has an "equality" operator of relu_abs(read_value - expected_value) where `relu_abs(x)=relu(x) + relu(-x)` 
	- Vocab embedding width of 64 floats
	- Ablation analysis:
		- Read output module made big difference:
			- 97.7%: `read_data - dense(in_signal, 128)` then relu_abs 
			- 86%: `dense(read_data, width=128)` then tanh
			- XX%: `dense(read_data, width=128)` then mi_activation
			- XX%: read_data - dense(in_signal, 128) then plain relu relu(x) performs worse (experiment currently running)
			- XX%: read_data then relu_abs, i.e. without subtraction of in_signal, performs worse (experiment currently running)
		- 97%: Having/removing indicator row (e.g. row just containing vocab unknown token) in edges DB made no difference
		- Output cell activation made big difference:
			- 50%: tanh (e.g. no better than random)
			- 97%: mi_activation
		- Read dropout causes slower convergence but same max accuracy



## Notes on training infrastructure

- FloydHub seems to fail with static decoding, need to do dynamic
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
- I believe the variation in dynamic decoding times is whether the network decides to do 1 or 2 iterations based on its finished flag.


