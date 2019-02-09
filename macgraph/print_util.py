
import tensorflow as tf
import numpy as np

from colored import fg, bg, stylize
from .const import EPSILON
import math
import networkx as nx


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


def normalize_levels(text_array, l):
	l_max = np.amax(l)
	l_min = np.amin(l)
	l_max = max(l_max, 1.0)

	print(f"min:{l_min}  max:{l_max}")

	l = (l - l_min) / (l_max + EPSILON)
	l = np.maximum(0.0, np.minimum(1.0, l))

	return l


def color_text(text_array, levels, color_fg=True):
	out = []

	levels = normalize_levels(text_array, levels)

	for l, s in zip(levels, text_array):
		if color_fg:
			color = fg(int(math.floor(DARK_GREY + l * (WHITE-DARK_GREY))))
		else:
			color = bg(int(math.floor(BG_BLACK + l * (BG_DARK_GREY-BG_BLACK))))
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

def measure_paths(row, vocab, node_from, node_to, avoiding, avoiding_property=1):

	graph = nx.Graph()

	def l(j):
		return vocab.inverse_lookup(j)

	only_using_nodes = [l(i[0]) for i in row["kb_nodes"] if l(i[avoiding_property]) != avoiding] + [node_from, node_to]
		
	for i in row["kb_nodes"]:
		graph.add_node( l(i[0]), attr_dict={"body": [l(j) for j in i]} )

	for id_a, connections in enumerate(row["kb_adjacency"][:row["kb_nodes_len"]]):
		for id_b, connected in enumerate(connections[:row["kb_nodes_len"]]):
			if connected:
				node_a = row["kb_nodes"][id_a]
				node_b = row["kb_nodes"][id_b]
				edge = (l(node_a[0]), l(node_b[0]))
				graph.add_edge(*edge)

	induced_subgraph = nx.induced_subgraph(graph, only_using_nodes)

	try:
		shortest_path_avoiding = len(nx.shortest_path(induced_subgraph, node_from, node_to))-2
	except:
		shortest_path_avoiding = None
		pass

	try:
		shortest_path = len(nx.shortest_path(graph, node_from, node_to)) -2
	except:
		shortest_path = None
		pass

	return {
		"shortest_path": shortest_path,
		"shortest_path_avoiding": shortest_path_avoiding,
	}


def adj_pretty(mtx, kb_nodes_len, kb_nodes, vocab):
	output = ""

	for r_idx, row in enumerate(mtx):
		if r_idx < kb_nodes_len:
			
			r_id = kb_nodes[r_idx][0]
			r_clean = vocab.inverse_lookup(kb_nodes[r_idx][1])
			r_name = vocab.inverse_lookup(r_id)
			output += pad_str(f"{r_name} - {r_clean}: ",target=20)
			
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