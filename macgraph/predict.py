
import tensorflow as tf
import numpy as np
from collections import Counter
from colored import fg, bg, stylize
import math
import argparse
import yaml
import os.path

from .input.text_util import UNK_ID
from .estimator import get_estimator
from .input import *
from .const import EPSILON


import logging
logger = logging.getLogger(__name__)

# Block annoying warnings
def hr():
	print(stylize("---------------", fg("blue")))

DARK_GREY = 235
WHITE = 255

BG_BLACK = 232
BG_DARK_GREY = 237

ATTN_THRESHOLD = 0.3

np.set_printoptions(precision=3)


def color_text(text_array, levels, color_fg=True):
	out = []
	for l, s in zip(levels, text_array):
		if color_fg:
			color = fg(int(math.floor(DARK_GREY + l*(WHITE-DARK_GREY))))
		else:
			color = bg(int(math.floor(BG_BLACK + l*(BG_DARK_GREY-BG_BLACK))))
		out.append(stylize(s, color))
	return out

def color_vector(vec, show_numbers=False):
	v_max = np.amax(vec)
	v_min = np.amin(vec)
	delta = np.abs(v_max - v_min)
	norm = (vec - v_min) / np.maximum(delta, 0.00001)

	def format_element(n):
		if show_numbers:
			return str(np.around(n, 4))
		else:
			return "-" if n < -EPSILON else ("+" if n > EPSILON else "0")

	def to_color(row):
		return ''.join(color_text([format_element(i) for i in row], (row-v_min) / np.maximum(delta, EPSILON)))
	
	return ' '.join(to_color(row) for row in vec)

def pad_str(s, target=3):
	if len(s) < target:
		for i in range(target - len(s)):
			s += " "
	return s

def adj_pretty(mtx, kb_nodes_len, kb_nodes, vocab):
	output = ""

	for r_idx, row in enumerate(mtx):
		if r_idx < kb_nodes_len:
			
			r_id = kb_nodes[r_idx][0]
			r_name = vocab.inverse_lookup(r_id)
			output += pad_str(f"{r_name}: ",target=4)
			
			for c_idx, item in enumerate(row):
				if c_idx < kb_nodes_len:

					c_id = kb_nodes[c_idx][0]
					c_name = vocab.inverse_lookup(c_id)

					if item:
						output += pad_str(f"{c_name}")
					else:
						output += pad_str(" ")
			output += "\n"

	return output


def predict(args, cmd_args):
	estimator = get_estimator(args)

	# Logging setup
	logging.basicConfig()
	tf.logging.set_verbosity("WARN")
	logger.setLevel("WARN")
	logging.getLogger("mac-graph").setLevel("WARN")

	# Info about the experiment, for the record
	tfr_size = sum(1 for _ in tf.python_io.tf_record_iterator(args["predict_input_path"]))
	logger.info(args)
	logger.info(f"Predicting on {tfr_size} input records")

	# Actually do some work
	predictions = estimator.predict(input_fn=gen_input_fn(args, "predict"))
	vocab = Vocab.load_from_args(args)

	def print_row(row):
		if p["actual_label"] == p["predicted_label"]:
			emoji = "✅"
			answer_part = f"{stylize(row['predicted_label'], bg(22))}"
		else:
			emoji = "❌"
			answer_part = f"{stylize(row['predicted_label'], bg(1))}, expected {row['actual_label']}"

		print("Finished:", color_vector(row['finished']))

		iterations = len(row["question_word_attn"])

		print(emoji, " ", answer_part)

		for i in range(iterations):

			# finished = row['finished'][i]
			# if finished:
			# 	print (f"{i}: Finished")
			
			if args["use_control_cell"]:
				for control_head in row["question_word_attn"][i]:
					print(f"{i}: " + ' '.join(color_text(row["src"], control_head)))

				# print(f"{i}: question_word_attn_raw: ", row["question_word_attn_raw"][i])
				# print(f"{i}: question_word_attn: ",     row["question_word_attn"][i])
			
			if args["use_read_cell"]:

				if len(args["kb_list"]) > 0:
					read_head_part = ' '.join(color_text(args["kb_list"], row["read_head_attn"][i]))
					print(f"{i}: read_head_attn: ",read_head_part)
				# print(f"{i}: read_attn_focus: ", row["read_head_attn_focus"][i])

				for idx0, noun in enumerate(args["kb_list"]):
					if row["read_head_attn"][i][idx0] > ATTN_THRESHOLD:
						db = [vocab.prediction_value_to_string(kb_row) for kb_row in row[f"{noun}s"] if kb_row[0] != UNK_ID]
						print(f"{i}: " + noun+"_attn: ",', '.join(color_text(db, row[f"{noun}_attn"][i])))

						for idx, attn in enumerate(row[f"{noun}_attn"][i]):
							if attn > ATTN_THRESHOLD:
								print(f"{i}: " +noun+"_word_attn: ",', '.join(color_text(
									vocab.prediction_value_to_string(row[f"{noun}s"][idx], True),
									row[f"{noun}_word_attn"][i],
									)
								))
			if args["use_message_passing"]:
				for tap in ["mp_read_attn", "mp_write_attn"]:
					db = [vocab.prediction_value_to_string(kb_row[0:1]) for kb_row in row["kb_nodes"]]
					db = db[0:row["kb_nodes_len"]]
					print(f"{i}: {tap}: ",', '.join(color_text(db, row[tap][i])))

				print(f"{i}: mp_write_signal: {row['mp_write_signal'][i]}")
				print(f"{i}: mp_read0_signal: {row['mp_read0_signal'][i]}")
				print(f"{i}: mp_node_state:   {color_vector(row['mp_node_state'][i][0:row['kb_nodes_len']])}")


			# if finished:
			# 	print("--FINISHED--")
			# 	break

		if args["use_message_passing"]:
			print("Adjacency:\n",
				adj_pretty(row["kb_adjacency"], row["kb_nodes_len"], row["kb_nodes"], vocab))

		hr()

	def decode_row(row):
		for i in ["type_string", "actual_label", "predicted_label", "src"]:
			row[i] = vocab.prediction_value_to_string(row[i], True)

	stats = Counter()
	output_classes = Counter()
	predicted_classes = Counter()
	confusion = Counter()

	for count, p in enumerate(predictions):
		if count >= cmd_args["n_rows"]:
			break

		decode_row(p)
		if cmd_args["filter_type_prefix"] is None or p["type_string"].startswith(cmd_args["filter_type_prefix"]):

			output_classes[p["actual_label"]] += 1
			predicted_classes[p["predicted_label"]] += 1

			correct = p["actual_label"] == p["predicted_label"]

			if correct:
				emoji = "✅"
			else:
				emoji = "❌"

			confusion[emoji + " \texp:" + p["actual_label"] +" \tact:" + p["predicted_label"] + " \t" + p["type_string"]] += 1

			if cmd_args["failed_only"] and not correct:
				print_row(p)
			elif cmd_args["correct_only"] and correct:
				print_row(p)
			elif not cmd_args["failed_only"] and not cmd_args["correct_only"]:
				print_row(p)


if __name__ == "__main__":
	tf.logging.set_verbosity(tf.logging.WARN)

	parser = argparse.ArgumentParser()
	parser.add_argument("--n-rows",type=int,default=20)
	parser.add_argument("--filter-type-prefix",type=str,default=None)
	parser.add_argument("--model-dir",type=str,required=True)
	parser.add_argument("--correct-only",action='store_true')
	parser.add_argument("--failed-only",action='store_true')


	cmd_args = vars(parser.parse_args())

	with tf.gfile.GFile(os.path.join(cmd_args["model_dir"], "config.yaml"), "r") as file:
		frozen_args = yaml.load(file)

	# If the directory got renamed, the model_dir might be out of sync, convenience hack
	frozen_args["model_dir"] = cmd_args["model_dir"]

	predict(frozen_args, cmd_args)



