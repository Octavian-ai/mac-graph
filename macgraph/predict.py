
import tensorflow as tf
import numpy as np
from collections import Counter
from colored import fg, bg, stylize
import math

from .input.text_util import UNK_ID
from .args import get_args
from .estimator import get_estimator
from .input import *

# Block annoying warnings
def hr():
	print(stylize("---------------", fg("blue")))

hr()

DARK_GREY = 242
WHITE = 256

BG_BLACK = 232
BG_DARK_GREY = 237

ATTN_THRESHOLD = 0.4

def extend_args(parser):
	parser.add_argument("--n-detail-rows",type=int,default=15)

def color_text(text_array, levels, color_fg=True):
	out = []
	for l, s in zip(levels, text_array):
		if color_fg:
			color = fg(int(math.floor(DARK_GREY + l*(WHITE-DARK_GREY))))
		else:
			color = bg(int(math.floor(BG_BLACK + l*(BG_DARK_GREY-BG_BLACK))))
		out.append(stylize(s, color))
	return out



def predict(args):
	estimator = get_estimator(args)
	predictions = estimator.predict(input_fn=gen_input_fn(args, "predict"))
	vocab = Vocab.load(args)

	def print_row(row):
		if p["actual_label"] == p["predicted_label"]:
			emoji = "✅"
			answer_part = f"{stylize(row['predicted_label'], bg(22))}"
		else:
			emoji = "❌"
			answer_part = f"{stylize(row['predicted_label'], bg(1))}, expected {row['actual_label']}"

		iterations = len(row["question_word_attn"])

		for i in range(iterations):
			print(emoji, ' '.join(color_text(row["src"].split(' '), row["question_word_attn"][i])), " ", answer_part, "\t")
			read_head_part = ' '.join(color_text(["nodes","edges"], row["read_head_attn"][i]))
			print("read_head_attn: ",read_head_part)
			for idx0, noun in enumerate(["node", "edge"]):
				# if row["read_head_attn"][i][idx0] > ATTN_THRESHOLD:
				db = [vocab.prediction_value_to_string(kb_row) for kb_row in row[f"kb_{noun}s"] if kb_row[0] != UNK_ID]
				print(noun+"_attn: ",', '.join(color_text(db, row[f"kb_{noun}_attn"][i])))

				for idx, attn in enumerate(row[f"kb_{noun}_attn"][i]):
					if attn > ATTN_THRESHOLD:
						print(noun+"_word_attn: ",', '.join(color_text(
							vocab.prediction_value_to_string(row[f"kb_{noun}s"][idx]).split(' '),
							row[f"kb_{noun}_word_attn"][i]))
						)

		hr()

	def decode_row(row):
		for i in ["type_string", "actual_label", "predicted_label", "src"]:
			row[i] = vocab.prediction_value_to_string(row[i])

	stats = Counter()
	output_classes = Counter()
	predicted_classes = Counter()
	confusion = Counter()

	for count, p in enumerate(predictions):
		decode_row(p)
		if args["type_string_prefix"] is None or p["type_string"].startswith(args["type_string_prefix"]):

			output_classes[p["actual_label"]] += 1
			predicted_classes[p["predicted_label"]] += 1

			if p["actual_label"] == p["predicted_label"]:
				emoji = "✅"
			else:
				emoji = "❌"

			confusion[emoji + " \texp:" + p["actual_label"] +" \tact:" + p["predicted_label"] + " \t" + p["type_string"]] += 1

			if count <= args["n_detail_rows"]:
				print_row(p)
			else:
				break

	# print(f"\nConfusion matrix:")
	# for k, v in confusion.most_common():
	# 	print(f"{k}: {v}")



if __name__ == "__main__":
	tf.logging.set_verbosity(tf.logging.WARN)
	args = get_args(extend_args)
	predict(args)



