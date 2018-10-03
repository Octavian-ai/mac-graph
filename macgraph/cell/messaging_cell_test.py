import unittest

import tensorflow as tf
tf.enable_eager_execution()

import numpy as np
import math

from .messaging_cell import *

class TestMessagingCell(unittest.TestCase):

    def test_basic_write(self):

        args = {
            "mp_state_width": 4,
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
            vocab_embedding[i][(i+1) % args["embed_width"]] = 1.0


        kb_nodes = np.tile(
            np.expand_dims(np.array(range(args["kb_nodes_len"])), 1),
            [1, args["kb_node_width"]]
        )

        features = {
            "kb_adjacency": batchify(np.eye(args["kb_nodes_len"])),
            "kb_nodes_len": batchify(args["kb_nodes_len"]),
            "d_batch_size": args["batch_size"],
            "kb_nodes": batchify( kb_nodes ),
        }



        table, table_width, table_len = get_table_with_embedding(args, features, vocab_embedding, "kb_node")

        # Mock queries
        in_write_query  = batchify(np.concatenate([vocab_embedding[0], np.zeros(args["embed_width"])], axis=-1))
        in_write_signal = batchify(np.random.random(args["mp_state_width"]))
        in_write_zero_signal = batchify(np.zeros(args["mp_state_width"]))
        in_read_query   = batchify(np.concatenate([vocab_embedding[2], np.zeros(args["embed_width"])], axis=-1))

        in_node_state = batchify(np.zeros([args["kb_nodes_len"], args["mp_state_width"]]))

        # --------------------------------------------------------------------------
        # Do two steps of propagation
        # --------------------------------------------------------------------------

        out_read_signal, out_node_state, taps = do_messaging_cell(args, features, vocab_embedding, 
            in_node_state, 
            table, table_width, table_len,
            in_write_query, in_write_signal, in_read_query)

        out_read_signal, out_node_state, taps = do_messaging_cell(args, features, vocab_embedding, 
            out_node_state, 
            table, table_width, table_len,
            in_write_query, in_write_zero_signal, in_read_query)

        rescale = in_write_signal[0,0] / out_read_signal[0,0] 
        rescaled = out_read_signal * rescale

        # print("in_write_query", in_write_query)
        # print("in_write_signal", in_write_signal)
        # print("in_read_query", in_read_query)
        # print("out_read_signal", rescaled)
        # print("out_node_state", out_node_state)
        print("rescale factor", rescale)
        # print("mp_write_scores", taps["mp_write_scores"])

        np.testing.assert_almost_equal(rescaled, in_write_signal, decimal=1)





if __name__ == '__main__':
    unittest.main()