import unittest

import tensorflow as tf
import numpy as np
import math

from .input import input_fn
from .build import build
from .args import get_args

class TestBuild(unittest.TestCase):

    def setUp(self):
        tf.enable_eager_execution()

    def assert_adjacency_valid(self, args, features, batch_index):

        lhs = features["kb_adjacency"][batch_index].numpy()

        # Test adjacency is symmetric
        np.testing.assert_array_equal(lhs, np.transpose(lhs))

        # Test not reflexive
        reflexive = np.identity(args["kb_node_max_len"], np.bool)
        any_reflexive = np.logical_and(reflexive, lhs)
        np.testing.assert_array_equal(any_reflexive, False)


        # Next, reconstruct adj from edges and check it is the same
        adj = np.full([args["kb_node_max_len"], args["kb_node_max_len"]], False)

        def get_node_idx(node_id):
            for idx, data in enumerate(features["kb_nodes"][batch_index]):
                if idx < features["kb_nodes_len"][batch_index].numpy():
                    if data[0].numpy() == node_id.numpy():
                        return idx

            raise ValueError(f"Node id {node_id} not found in node list {features['kb_nodes'][batch_index]}")

        for idx, edge in enumerate(features["kb_edges"][batch_index]):
            if idx < features["kb_edges_len"][batch_index].numpy():
                node_from = edge[0]
                node_to = edge[2]
                node_from_idx = get_node_idx(node_from)
                node_to_idx = get_node_idx(node_to)
                adj[node_from_idx][node_to_idx] = True
                adj[node_to_idx][node_from_idx] = True

        assert adj.shape == features["kb_adjacency"][batch_index].numpy().shape


        # So that it prints useful errors
        np.set_printoptions(threshold=np.inf)
        # for i, j, k in zip(lhs, adj, features["kb_nodes"][batch_index]):
        #     print("testing node", k.numpy())
        #     np.testing.assert_array_equal(i,j)
        np.testing.assert_array_equal(adj, lhs)
        

    def test_build_adjacency(self):

        argv = [
            '--gqa-path',  'input_data/raw/gqa-test.yaml',
            '--input-dir', 'input_data/processed/test',
            '--limit', '100',
            '--predict-holdback', '0',
            '--eval-holdback', '0',
        ]

        args = get_args(argv=argv)
        build(args)
        dataset = input_fn(args, "train", repeat=False)

        for features, label in dataset:
            for i in range(len(list(label))):
                self.assert_adjacency_valid(args, features, i)





if __name__ == '__main__':
    unittest.main()