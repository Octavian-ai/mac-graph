import unittest

import tensorflow as tf
tf.enable_eager_execution()

import numpy as np
import math
import random

from .messaging_cell import *

class TestMessagingCell(unittest.TestCase):

    def test_basic_write(self):

        args = {
            "mp_state_width": 2,
            "kb_node_max_len": 23,
            "mp_activation": "linear",
            "kb_nodes_len": 23,
            "kb_node_width": 2,
            "read_indicator_rows": 0,
            "embed_width": 32,
            "batch_size": 1,
        }

        def batchify(a):
            a = np.array(a)
            dims = np.ones(a.ndim+1)
            dims[0] = args["batch_size"]
            return np.tile(np.array(a), dims)


        vocab_embedding = np.zeros([args["kb_nodes_len"], args["embed_width"]], float)
        for i in range(args["kb_nodes_len"]):
            vocab_embedding[i][(i) % args["embed_width"]] = 1.0

        kb_adjacency = np.zeros([args["kb_nodes_len"], args["kb_nodes_len"]])
        for i in range(args["kb_nodes_len"]):
            connections_to = [i+1, i+7]
            for to in connections_to:
                kb_adjacency[i][(to) % args["kb_nodes_len"]] = 1.0

        kb_nodes = np.tile(
            np.expand_dims(np.array(range(args["kb_nodes_len"])), 1),
            [1, args["kb_node_width"]]
        )

        features = {
            "kb_adjacency": batchify(kb_adjacency),
            "kb_nodes_len": batchify(args["kb_nodes_len"]),
            "d_batch_size": args["batch_size"],
            "kb_nodes": batchify( kb_nodes ),
        }



        table, table_width, table_len = get_table_with_embedding(args, features, vocab_embedding, "kb_node")

        # Mock queries
        in_write_query           = batchify(np.concatenate([vocab_embedding[0], np.zeros(args["embed_width"])], axis=-1))
        in_write_signal1         = batchify([10.0,0.0])
        in_write_signal2         = batchify([0.0,10.0])
        in_write_zero_signal     = batchify(np.zeros(args["mp_state_width"]))
        in_read_query            = batchify(np.concatenate([vocab_embedding[2], np.zeros(args["embed_width"])], axis=-1))
        in_read_query_elsewhere  = batchify(np.concatenate([vocab_embedding[4], np.zeros(args["embed_width"])], axis=-1))
        SIGNAL_MIN_DELTA = 0.1

        # print("in_write_signal", in_write_signal1)


        node_state = batchify(np.ones([args["kb_nodes_len"], args["mp_state_width"]]))

        # --------------------------------------------------------------------------
        # Do two steps of propagation and check the message arrives
        # --------------------------------------------------------------------------

        out_read_signals, node_state, taps = do_messaging_cell(args, features, vocab_embedding, 
            node_state, 
            table, table_width, table_len,
            in_write_query, in_write_signal1, [])

        # print("node_state", node_state)

        out_read_signals, node_state, taps = do_messaging_cell(args, features, vocab_embedding, 
            node_state, 
            table, table_width, table_len,
            in_write_query, in_write_signal2, [in_read_query, in_read_query_elsewhere])

        # print("node_state", node_state)

        # rescale = in_write_signal1[0,0] / out_read_signals[0][0,0] 
        # rescaled_read = out_read_signals[0] * rescale
        # print("out_read_signal", out_read_signals)
        # print("rescale factor", rescale)
        # print("rescaled_read1", rescaled_read)

        # np.testing.assert_almost_equal(rescaled_read, in_write_signal1, decimal=3)

        print("ON:",out_read_signals[0][0][0])
        print("OFF:",out_read_signals[1][0][0])
       

        self.assertGreater(out_read_signals[0][0][0], out_read_signals[1][0][0])


        # --------------------------------------------------------------------------
        #  check the message doesn't arrive to distant node
        # --------------------------------------------------------------------------

        # rescaled_read = out_read_signals[1] * rescale

        # print("node_state", node_state)
        # print("read_scores", taps["mp_read_attn"])

        # print("out_read_signal", out_read_signal)
        # print("rescale factor", rescale)
        # print("rescaled_read2", rescaled_read)

        # with self.assertRaises(Exception):
            # np.testing.assert_almost_equal(rescaled_read, in_write_signal1, decimal=3)





if __name__ == '__main__':
    unittest.main()