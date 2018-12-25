
import tensorflow as tf
import numpy as np

from colored import fg, bg, stylize
from .const import EPSILON
import math


TARGET_CHAR_WIDTH = 80

def hr_text(text):
	t_len = len(text) + 2
	pad_len = (TARGET_CHAR_WIDTH - t_len) // 2
	padding = '-'.join(["" for i in range(pad_len)])

	s = padding + " " + text + " " + padding

	print(stylize(s, fg("yellow")))


def hr(bold=False):
	if bold:
		print(stylize("--------------------------", fg("yellow")))
	else:
		print(stylize("--------------------------", fg("blue")))

DARK_GREY = 235
WHITE = 255

BG_BLACK = 232
BG_DARK_GREY = 237

ATTN_THRESHOLD = 0.25

np.set_printoptions(precision=3)


def color_text(text_array, levels, color_fg=True):
	out = []

	l_max = np.amax(levels)
	l_min = np.amin(levels)

	l_max = max(l_max, 1.0)

	for l, s in zip(levels, text_array):
		l_n = (l - l_min) / (l_max + EPSILON)
		l_n = max(0.0, min(1.0, l_n))
		if color_fg:
			color = fg(int(math.floor(DARK_GREY + l_n * (WHITE-DARK_GREY))))
		else:
			color = bg(int(math.floor(BG_BLACK + l_n * (BG_DARK_GREY-BG_BLACK))))
		out.append(stylize(s, color))
	return out

def color_vector(vec, show_numbers=True):

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
		return ' '.join(color_text([format_element(i) for i in row], (row-v_min) / np.maximum(delta, EPSILON)))
	
	if len(np.shape(vec)) == 1:
		return to_color(vec)
	else:
		return [to_color(row) for row in vec]

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