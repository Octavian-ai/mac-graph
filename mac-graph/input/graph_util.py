
import numpy as np

from .text_util import *

NODE_PROPS = ["name", "cleanliness"]
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

	def node_to_vec(node):
		return np.array([
			vocab.lookup(node[key]) for key in NODE_PROPS
		])

	def edge_to_vec(edge):
		return np.array([
			vocab.lookup(edge[key]) for key in EDGE_PROPS
		])

	def pack(row):
		if len(row) > args["kb_width"]:
			return row[0:args["kb_width"]]
		elif len(row) < args["kb_width"]:
			return np.pad(row, (0,args["kb_width"] - len(row)), 'constant', constant_values=0)
		else:
			return row

	edges = []

	node_lookup = {i["id"]: i for i in graph["nodes"]}

	nodes = [pack(node_to_vec(i)) for i in graph["nodes"]]

	for edge in graph["edges"]:
		s1 = node_to_vec(node_lookup[edge["station1"]])
		s2 = node_to_vec(node_lookup[edge["station2"]])
		e = edge_to_vec(edge)

		row = np.concatenate((s1, e, s2), -1)
		row = pack(row)
		
		assert len(row) == args["kb_width"], "Extraction functions didn't create the right length of knowledge table data"

		edges.append(row)

	return np.array(nodes), np.array(edges)

