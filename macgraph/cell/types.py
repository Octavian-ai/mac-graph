

from collections import NamedTuple
from typing import Dict

DigestedQuery = NamedTuple('DigestedQuery', ['fwd_state', 'bwd_state', 'token_states'])

class CellContext(NamedTuple):
	features: Dict
	args: Dict,
	vocab_embedding
	in_prev_outputs
	in_iter_id
	in_iter_question_state
	in_memory_state
	in_question_tokens
	in_question_state
	in_node_state