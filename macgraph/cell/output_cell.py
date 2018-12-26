
import tensorflow as tf

from ..minception import *
from ..args import ACTIVATION_FNS
from ..util import *
from ..layers import *
from ..attention import *
from ..component import *


class OutputCell(Component):

	def __init__(self, args):
		super().__init__(args, "output_cell")

		# TODO: generate this from components
		tr = []
		for i in range(args["mp_read_heads"]):
			tr.append(f"mp{i}")

		for i in range(args["max_decode_iterations"]):
			tr.append(f"po{i}")

		if args["use_question_state"]:
			tr.append("q_state")
			
		if args["use_memory_cell"]:
			tr.append("mem")
		
		if args["use_read_cell"]:
			for i in range(args["read_heads"]):
				tr.append(f"read{i}")



		self.output_table = Tensor("table")
		self.output_query = Tensor("focus_query")
		self.focus = AttentionByIndex(args, 
			self.output_table, self.output_query, seq_len=6, 
			table_representation=tr,
			name="focus")

		

	def forward(self, features, context, memory_state, reads, mp_reads):

		# TODO: Remove transitional scaffolding
		self.context = context
		self.memory_state = memory_state
		self.reads = reads
		self.mp_reads = mp_reads

		with tf.name_scope(self.name):

			in_all = []

			def add(t):
				in_all.append(pad_to_len_1d(t, self.args["input_width"]))

			def add_all(t):
				for i in t:
					add(i)

			if self.args["use_message_passing"]:
				add_all(self.mp_reads)

			prev_outputs = tf.unstack(self.context.in_prev_outputs, axis=1)
			add_all(prev_outputs)
			
			if self.args["use_question_state"]:
				add(self.context.in_question_state)

			if self.args["use_memory_cell"]:
				add(self.memory_state)
			
			if self.args["use_read_cell"]:
				add_all(self.reads)

			
			in_stack = tf.stack(in_all, axis=1)
			in_stack = dynamic_assert_shape(in_stack, [features["d_batch_size"], len(in_all), self.args["input_width"]])

			self.output_table.bind(in_stack)
			self.output_query.bind(context.in_iter_id)
			v = self.focus.forward(features)
			v.set_shape([None, self.args["input_width"]])

			for i in range(self.args["output_layers"]):
				v = layer_dense(v, self.args["output_width"], self.args["output_activation"], name=f"output{i}")

			return v



