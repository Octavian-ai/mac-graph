
import numpy as np

def graph_to_table(args, graph):

	def node_to_vec(node):
		return np.array([0,1,0])

	def edge_to_vec(edge):
		return np.array([1,0,0])


	table = []

	node_lookup = {i["id"]: i for i in graph["nodes"]}

	for edge in graph["edges"]:
		s1 = node_to_vec(node_lookup[edge["station1"]])
		s2 = node_to_vec(node_lookup[edge["station2"]])
		e = edge_to_vec(edge)

		row = np.concat((s1, e, s2), -1)
		row = np.pad(row, (0,args["kb_width"] - len(row)), 'constant', constant_values=0)
		assert len(row) == args["kb_width"], "Extraction functions didn't create the right length of knowledge table data"

		table.append(row)

	retur np.array(table)

