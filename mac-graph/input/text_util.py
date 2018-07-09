
from collections import Counter
import tensorflow as tf
from typing import List, Set
import re
import string

import logging
logger = logging.getLogger(__name__)

from .util import read_gqa



# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------


UNK = "<unk>"
SOS = "<sos>"
EOS = "<eos>"
SPACE = "<space>"

CHARS = ["<"+i+">" for i in string.ascii_lowercase] + ["<"+i+">" for i in string.ascii_uppercase]
SPECIAL_TOKENS = [UNK, SOS, EOS, SPACE] + CHARS

UNK_ID = SPECIAL_TOKENS.index(UNK)
SOS_ID = SPECIAL_TOKENS.index(SOS)
EOS_ID = SPECIAL_TOKENS.index(EOS)



# --------------------------------------------------------------------------
# Pretokenize
# --------------------------------------------------------------------------


ENGLISH_PUNCTUATION = '!"#$%&()*+,-./:;=?@[\\]^_`{|}~'

# --------------------------------------------------------------------------


def pretokenize_general(text):
	text = text.replace("\n", "")
	text = re.sub(r'\s*$', '', text)
	text = text.replace(" ", f" {SPACE} ")
	return text


def pretokenize_json(value):
	if isinstance(value, str) or isinstance(value, bool) or isinstance(value, int):
		return str(value)

	raise ValueError("Unsupported json value type")



def pretokenize_english(text):

	text = pretokenize_general(text)

	for p in ENGLISH_PUNCTUATION:
		text = text.replace(p, f" {p} ")

	return text




# --------------------------------------------------------------------------
# Vocab
# --------------------------------------------------------------------------


class Vocab(object):

	def __init__(self, table:List[str]):
		self.table = table

	def __contains__(self, value):
		return value in self.table

	def __iter__(self):
		return iter(self.table)

	# -------------------------------------------------------------------------- #

	def lookup(self, value):
		try:
			return self.table.index(value)
		except ValueError:
			return UNK_ID

	def ids_to_string(self, line):
		return ' '.join([self.table[i] for i in line])

	def string_to_ids(self, line):
		return [self.lookup(i) for i in line.split(' ')]

	def expand_unknowns(self, line):
		unknowns = set(line.split(' '))
		unknowns -= set(self.table)
		unknowns -= set([''])

		for t in unknowns:
			spaced = ''.join([f"<{c}> " for c in t])
			line = line.replace(t, spaced)

		return line


	def english_to_ids(self, line):
		# TODO: Make greedy w.r.t. tokens with spaces in them
		line = pretokenize_english(line)
		line = self.expand_unknowns(line)
		line = self.string_to_ids(line)
		return line



	def save(self, args):
		with tf.gfile.GFile(args["vocab_path"], 'w') as out_file:
			for i in self.table:
				out_file.write(i + "\n")



	# --------------------------------------------------------------------------
	# Make me a vocab!
	# --------------------------------------------------------------------------
	

	@classmethod
	def load(cls, args):

		tokens = list()

		with tf.gfile.GFile(args["vocab_path"]) as file:
			for line in file.readlines():
				tokens.append(line.replace("\n", ""))
				
				if len(tokens) == args["vocab_size"]:
					break

		assert len(tokens) == len(set(tokens)), f"Duplicate lines in {args['vocab_path']}"

		return Vocab(tokens)



	@classmethod
	def build(cls, args, gqa_to_tokens):
		hits = Counter()

		def add(tokens:List[str]):
			for token in tokens:
				if token not in ["", " ", "\n"]:
					hits[token] += 1

		for i in read_gqa(args):
			add(gqa_to_tokens(i))

		tokens = list()
		tokens.extend(SPECIAL_TOKENS)

		for i, c in hits.most_common(args["vocab_size"]):
			if len(tokens) == args["vocab_size"]:
				break

			if i not in tokens:
				tokens.append(i)

		assert len(tokens) <= args["vocab_size"]
		
		v = Vocab(tokens)
		v.save(args)

		return v




	