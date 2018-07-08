
import numpy as np

from .text_util import *

def graph_to_table(args, vocab, graph):

	def node_to_vec(node):
		node_props = ["name", "architecture", "cleanliness", "has_rail", "disabled_access", "size", "music"]

		return np.array([
			lookup_vocab(pretokenize_json(node[key]), vocab) for key in node_props
		])

	def edge_to_vec(edge):
		edge_props = ["line_name", "line_color"]
		return np.array([
			lookup_vocab(pretokenize_json(edge[key]), vocab) for key in edge_props
		])


	table = []

	node_lookup = {i["id"]: i for i in graph["nodes"]}

	for edge in graph["edges"]:
		s1 = node_to_vec(node_lookup[edge["station1"]])
		s2 = node_to_vec(node_lookup[edge["station2"]])
		e = edge_to_vec(edge)

		row = np.concatenate((s1, e, s2), -1)
		row = np.pad(row, (0,args["kb_width"] - len(row)), 'constant', constant_values=0)
		assert len(row) == args["kb_width"], "Extraction functions didn't create the right length of knowledge table data"

		table.append(row)

	return np.array(table)

