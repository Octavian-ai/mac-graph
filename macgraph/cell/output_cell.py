
import tensorflow as tf

from ..minception import *
from ..args import ACTIVATION_FNS
from ..util import *
from ..layers import *
from ..attention import *
from ..component import *


def OutputCell(Component):

	def __init__(context, memory_state, reads, mp_reads):
		self.context = context
		self.memory_state = memory_state
		self.reads = reads
		self.mp_reads = reads

		super.__init__("output_cell")

	def forward(self, args, features):

		with tf.name_scope(self.name):

			in_all = []
			
			if args["use_question_state"]:
				in_all.append(context.in_question_state)

			if args["use_memory_cell"]:
				in_all.append(in_memory_state)
			
			if args["use_output_read"]:
				in_all.extend(in_reads)

			if args["use_message_passing"]:
				in_all.extend(in_mp_reads)

			prev_query = layer_dense(context.in_iter_id, args["input_width"])

			self.from_prev = Attention(Tensor(context.in_prev_outputs), Tensor(prev_query))
			in_all.append(from_prev.forward(args, features))

			in_all = tf.concat(in_all, -1)

			v = in_all
			finished = in_all

			for i in range(args["output_layers"]):
				v = layer_dense(v, args["output_width"], args["output_activation"], name=f"output{i}")

			return v