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


# Best known args:
# {'log_level': 'DEBUG', 'output_dir': './output', 'input_dir': 'input_data/processed/sa_small_100k_balanced', 'model_dir': 'output/model/sa/956b4f6', 'limit': None, 'type_string_prefix': None, 'eval_holdback': 0.1, 'predict_holdback': 0.005, 'warm_start_dir': None, 'batch_size': 32, 'max_steps': None, 'max_gradient_norm': 4.0, 'learning_rate': 0.001, 'dropout': 0.2, 'answer_classes': 8, 'vocab_size': 90, 'embed_width': 32, 'num_input_layers': 3, 'kb_node_width': 7, 'kb_edge_width': 3, 'read_heads': 1, 'data_stack_width': 64, 'data_stack_len': 20, 'control_width': 64, 'memory_width': 128, 'memory_transform_layers': 2, 'use_kb_node': False, 'use_kb_edge': True, 'use_data_stack': False, 'use_control_cell': True, 'use_dynamic_decode': True, 'max_decode_iterations': 2, 'modes': ['eval', 'train', 'predict'], 'eval_input_path': 'input_data/processed/sa_small_100k_balanced/eval_input.tfrecords', 'train_input_path': 'input_data/processed/sa_small_100k_balanced/train_input.tfrecords', 'predict_input_path': 'input_data/processed/sa_small_100k_balanced/predict_input.tfrecords', 'all_input_path': 'input_data/processed/sa_small_100k_balanced/all_input.tfrecords', 'vocab_path': 'input_data/processed/sa_small_100k_balanced/vocab.txt', 'question_types_path': 'input_data/processed/sa_small_100k_balanced/types.yaml', 'answer_classes_path': 'input_data/processed/sa_small_100k_balanced/answer_classes.yaml'}