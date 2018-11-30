import unittest

import tensorflow as tf
import numpy as np
import math
import logging

from .input import input_fn
from .build import build
from .args import get_args
from .util import *

logger = logging.getLogger(__name__)

class TestBuild(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        tf.enable_eager_execution()

        argv = [
            '--gqa-paths',  'input_data/raw/test.yaml',
            '--input-dir', 'input_data/processed/test',
            '--limit', '100',
            '--predict-holdback', '0.1',
            '--eval-holdback', '0.1',
        ]

        args = get_args(argv=argv)
        cls.args = args

        build(args)

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

        dataset = input_fn(TestBuild.args, "train", repeat=False)

        for features, label in dataset:
            for i in range(len(list(label))):
                self.assert_adjacency_valid(TestBuild.args, features, i)


    def test_build_basics(self):

        # Validate questions are a unique set
        gqa_questions = set()
        for i in read_gqa(TestBuild.args):
            digest = (i["question"]["english"], len(i["graph"]["edges"]), len(i["graph"]["nodes"]))
            self.assertNotIn(digest, gqa_questions)
            gqa_questions.add(digest)

        questions = {}

        for mode in TestBuild.args["modes"]:
            dataset = input_fn(TestBuild.args, mode, repeat=False)
            questions[mode] = set()

            for features, label in dataset:
                for batch_index in range(len(list(label))):
                    questions[mode].add((
                        str(features["src"][batch_index]), int(features["kb_edges_len"][batch_index]), int(features["kb_nodes_len"][batch_index])
                    ))


        for mode in TestBuild.args["modes"]:
            for mode_b in TestBuild.args["modes"]:
                if mode != mode_b:
                    self.assertTrue(questions[mode].isdisjoint(questions[mode_b]), f"Same question in mode {mode} and {mode_b}")
                   






if __name__ == '__main__':
    logging.basicConfig()
    logger.setLevel('INFO')

    unittest.main()