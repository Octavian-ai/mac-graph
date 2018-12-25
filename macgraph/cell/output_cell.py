
import tensorflow as tf

from ..minception import *
from ..args import ACTIVATION_FNS
from ..util import *
from ..layers import *
from ..attention import *
from ..component import *


class OutputCell(Component):

	def __init__(self, args):
		self.in_prev_outputs = Tensor("in_prev_outputs")
		self.prev_query = Tensor("prev_query")
		self.from_prev = Attention(args, self.in_prev_outputs, self.prev_query, seq_len=args["max_decode_iterations"], name="from_prev")

		super().__init__(args, "output_cell")

	def forward(self, features, context, memory_state, reads, mp_reads):

		# TODO: Remove transitional scaffolding
		self.context = context
		self.memory_state = memory_state
		self.reads = reads
		self.mp_reads = mp_reads

		with tf.name_scope(self.name):

			in_all = []
			
			if self.args["use_question_state"]:
				in_all.append(self.context.in_question_state)

			if self.args["use_memory_cell"]:
				in_all.append(self.memory_state)
			
			if self.args["use_output_read"]:
				in_all.extend(self.reads)

			if self.args["use_message_passing"]:
				in_all.extend(self.mp_reads)

			self.in_prev_outputs.bind(self.context.in_prev_outputs)
			self.prev_query.bind(layer_dense(self.context.in_iter_id, self.args["input_width"]))

			in_all.append(self.from_prev.forward(features))

			in_all = tf.concat(in_all, -1)

			v = in_all
			finished = in_all

			for i in range(self.args["output_layers"]):
				v = layer_dense(v, self.args["output_width"], self.args["output_activation"], name=f"output{i}")

			return v



