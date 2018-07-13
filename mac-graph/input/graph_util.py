
import numpy as np

from .text_util import *

NODE_PROPS = ["name", "cleanliness", "music", "architecture", "size", "has_rail", "disabled_access"]
EDGE_PROPS = ["line_name"]


def gqa_to_tokens(args, gqa):

	tokens = list()

	for edge in gqa["graph"]["edges"]:
		for key in EDGE_PROPS:
			tokens.append(pretokenize_json(edge[key]))

	for node in gqa["graph"]["nodes"]:
		for key in NODE_PROPS:
			tokens.append(pretokenize_json(node[key]))

	tokens += pretokenize_english(gqa["question"]["english"]).split(' ')

	try:
		tokens.append(pretokenize_json(gqa["answer"]))
	except ValueError:
		pass

	return tokens


def graph_to_table(args, vocab, graph):

	def node_to_vec(node, props=NODE_PROPS):
		return np.array([
			vocab.lookup(pretokenize_json(node[key])) for key in props
		])

	def edge_to_vec(edge, props=EDGE_PROPS):
		return np.array([
			vocab.lookup(pretokenize_json(edge[key])) for key in props
		])

	def pack(row, width):
		if len(row) > width:
			r = row[0:width]
		elif len(row) < width:
			r = np.pad(row, (0, width - len(row)), 'constant', constant_values=UNK_ID)
		else:
			r = row

		assert len(r) == width, "Extraction functions didn't create the right length of knowledge table data"
		return r

	edges = []

	node_lookup = {i["id"]: i for i in graph["nodes"]}

	nodes = [pack(node_to_vec(i), args["kb_node_width"]) for i in graph["nodes"]]

	for edge in graph["edges"]:
		s1 = node_to_vec(node_lookup[edge["station1"]])
		s2 = node_to_vec(node_lookup[edge["station2"]])
		e = edge_to_vec(edge)

		row = np.concatenate((s1, e, s2), -1)
		row = pack(row, args["kb_node_width"])
		
		edges.append(row)

	return np.array(nodes), np.array(edges)

