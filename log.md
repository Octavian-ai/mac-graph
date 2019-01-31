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

- `636d354`: as above

- `e901f89`: 82% after an hour on Floyd

- `86804e6`: 
	- Trying to resurrect old results, got 80% @ 10k then plateau
	- Does well on all questions except HasRail, HasDisabledAccess

- `b894c56`: 98% @ 200k training steps (e.g. slow, but gets there)

- `1240a73`: 93% @ 50k steps, running on expanded "small" 100k dataset
	- added node-token attention to the retrieved row, helped a lot

- `9d298d7`: 92% @ 36k - extracting row token by position seems helpful
- `53f5e0f`: 93% @ 40k - fastest (wall) to reach accuracy (reduced cell iterations)
- `572d98e`: 99% @ 150k - woop. Disabled memory, down to single step
- `31fd026`: 100% @ 6k steps - added residual to first layer of input biLSTM and made that wider


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
			- 77%: `dense(read_data, width=128)` then mi_activation
			- 67%: `read_data - dense(in_signal, 128)` then plain relu relu(x) 
			- 97.7%: read_data then relu_abs, i.e. without subtraction of in_signal
		- 97%: Having/removing indicator row (e.g. row just containing vocab unknown token) in edges DB made no difference
		- Output cell activation made big difference:
			- 50%: tanh (e.g. no better than random)
			- 97%: mi_activation
		- Read dropout causes slower convergence but same max accuracy

	- Restoring rest of network:
		- 96.5%: baseline + read dropout, dynamic decode, question state (2hrs)

- `2d434c1`: 
	- currently exeriences nan loss at times from softmax
	- softmax args are more than 0 despite subtracting max

- `e901d46`: Initial training same trajectory as `d503000`
	- `4c31b1b`, `fec44c7` same
	- May plateau out later, unsure

- `dabccfe`: Switched from absu to mi_act and tanked at 66% accuracy
- `e5585e1`: 100k, 93%, seems to be starting to plateau
- `63d1002`: Add read indi row, faster increase of accuracy 95% @ 120k, 96% @400k
- `e1bdfe9`: 97.6% @ 114k (train-station-dependency.sh)
- `4b57eed` 90% @ 15k (train-experimental.sh) by accident
- `33d76f9`: 95% @ 23k best ever (-experimental)
- `bfd04cd` max gradient norm 0.4 allows for correctly sized vocab embedding variable. 97% @ 90k

- `dd76eb6`: 100% acc @ 30k with message passing

- `468597b`: 100% accuracy with message passing


### Combined station adjacency and properties

- `d5f2dd3` 97.5% @ 140k steps 
	- word-attention system really helped
	- tried max-pooling tanh and abs as the read activation function
	- still struggling with True/False answers, StationAdjacent 73% @ 140k, StationPropertyArchitecture 93%

- `bb3ff8e` 97% @ 20k steps
	- struggling on True/False
	- adjacency 78%
	- disabled access 50%
	- has rail 60%

- `e6540b7` 88% @ 17k
	- 78% on station adjacency
	- 99% on all station property questions
	- I believe I need to give the network a way to choose read transform based on control state

- `230ab8a` 97% @ 17k
	- 88% on station adjacency
	- 99% on all station property questions

- `6d5eb35` 99% on just station adjacency (1 read layer and read act tanh_abs helped immensely)
- `a427516` 99% on just station adjacency (3 read layers)
	- this proves that the read activation fn is the critical part to succesfully solving SA

- `4ff3692` 99% on just station properties (using same net as previous successful 99% SA run)

- `9e7ab2d` 99% on SP, 85% on SA (70k steps)
	- net known to get 99% on each task individually
	- net gets to 99% on SA alone after 5k steps

- `8064864` Running same net against SA, SP, SA+SP with summaries shows that
	SA choses to use the edge head, and SP choses to use the node head. SA+SP
	choses to use the node head.

- `63a8a8b` 99% on all tasks
	- added dedicated control heads x2 for each read head
	- `use_read_query_block` didn't seem to help

### Station one apart
- `44563f0`: Message passing 99% accuracy @ 10k steps

### Station zero/one/two apart
- `0bf8504`: Message passing 97.6% accuracy @ 4k steps

### Station shortest count

- `9ecf10e` 
	- Feeding memory into the read_cell was hampering performance
	- Idea to use softmax selection of read input (and possibly activation) to allow specialisation of read heads

 - `ce9ca1a70fc678e3d32479888332d322007a1bb2`
 	- 100% on adjacency with forced query

 - `9866573` 100% with forced write query and token_index read query
 	- Interesting because using an attn mix of token_index and token_content failed

 - `44972f3` 
 	- sometimes works for 0 to 1 paths - it gets stuck on read + write of same node at times
 	- --fast causes it to fail, otherwise it succeeds

 - `4bef897`
 	- works for iter two after shrinking LR to 0.1
 	- single mp head, output cell has attention_by_index of both mp reads and previous outputs

 - `ee22e45`: full dataset, 40 iter, up to 99% accuracy

 - `9a881d7`: full dataset up to 9, 10 iteration 99.6% - breakthrough was using state width of 32 so
 		the network can send different elements at different times for timing


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


