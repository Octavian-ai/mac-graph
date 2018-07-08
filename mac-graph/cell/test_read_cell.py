
import tensorflow as tf
import numpy as np


from .read_cell import *




class ReadTest(tf.test.TestCase):
	
	def test_basic_read(self):

		in_content = [1,0,0]
		in_mask = [1,0,0]
		in_knowledge = np.array([
			[1,1,1],
			[0,0,0]
		])
		out_expected =  0.731059 * in_knowledge[0] + (1 - 0.731059) * in_knowledge[1]

		args = {
			"batch_size": 1,
			"kb_width": 3,
			"kb_len": 2,
			"vocab_size": 3,
		}

		W_score = np.identity(3)
		vocab_embedding = tf.eye(args["vocab_size"])


		with self.test_session():
			in_content_p   = tf.placeholder(tf.float32, [args["batch_size"], args["kb_width"]])
			in_mask_p      = tf.placeholder(tf.float32, [args["batch_size"], args["kb_width"]])
			in_knowledge_p = tf.placeholder(tf.float32, [args["kb_len"], args["kb_width"]])
			W_score_p      = tf.placeholder(tf.float32, [args["kb_width"], args["kb_width"]])

			feed_dict = {
				in_content_p:   [in_content],
				in_mask_p:      [in_mask],
				in_knowledge_p: in_knowledge,
				W_score_p:		W_score,
			}

			out_data = read_from_graph(args, in_content_p, in_mask_p, in_knowledge_p, vocab_embedding, W_score=W_score_p)

			self.assertAllClose(out_data.eval(feed_dict), [out_expected])


if __name__ == '__main__':
	tf.test.main()