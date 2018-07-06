
from .read_cell import *
from .write_cell import *
from .control_cell import *

def mac_cell(args, in_states, in_question_state, in_question_tokens, in_knowledge):

	in_control_state, in_memory_state = in_states

	out_control_state = control_cell(args, in_control_state, in_question_state, in_question_tokens)
	data_read = read_cell(args, in_memory_state, out_control_state, in_knowledge)
	out_memory_state = write_cell(args, in_memory_state, data_read, out_control_state)

	return (out_control_state, out_memory_state)