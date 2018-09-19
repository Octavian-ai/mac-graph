
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

	assert len(graph["nodes"]) <= args["kb_node_max_len"]

	for edge in graph["edges"]:
		s1 = node_to_vec(node_lookup[edge["station1"]], ['name'])
		s2 = node_to_vec(node_lookup[edge["station2"]], ['name'])
		e  = edge_to_vec(edge)

		row = np.concatenate((s1, e, s2), -1)
		row = pack(row, args["kb_edge_width"])
		
		edges.append(row)


	# I'm treating edges as bidirectional for the adjacency matrix
	# Also, I'm discarding line information. That is still in the edges list 
	def is_connected(idx_from, idx_to):

		# To produce stable tensor sizes, the adj matrix is padded out to kb_nodes_max_len
		if idx_from >= len(graph["nodes"]) or idx_to >= len(graph["nodes"]):
			return False

		id_from = graph["nodes"][idx_from]["id"]
		id_to   = graph["nodes"][idx_to  ]["id"]

		for edge in graph["edges"]:
			if edge["station1"] == id_from and edge["station2"] == id_to:
				return True
			if edge["station1"] == id_to   and edge["station2"] == id_from:
				return True

		return False


	adjacency = [
		[
			is_connected(i, j) for j in range(args["kb_node_max_len"])
		]
		for i in range(args["kb_node_max_len"])
	]


	return np.array(nodes), np.array(edges), np.array(adjacency)

