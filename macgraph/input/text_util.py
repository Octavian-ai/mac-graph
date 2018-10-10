
from collections import Counter
import tensorflow as tf
import numpy as np
from typing import List, Set
import re
import string
from tqdm import tqdm

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
SPECIAL_TOKENS = [UNK, SOS, EOS, SPACE] #+ CHARS

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

def detokenize_general(text):
	text = text.replace(f" {SPACE} ", " ")
	return text


def pretokenize_json(value):
	if isinstance(value, str) or isinstance(value, bool) or isinstance(value, int):
		return str(value)
	raise ValueError("Unsupported json value type")


def pretokenize_english(text):
	text = pretokenize_general(text)

	for p in ENGLISH_PUNCTUATION:
		text = text.replace(p, f" {p} ")

	text = re.sub(r'\s*$', '', text)
	return text


def detokenize_english(text):
	text = detokenize_general(text)

	for p in ENGLISH_PUNCTUATION:
		text = text.replace(f" {p} ", p)

	return text


def bytes_to_string(p):
	if len(p) == 0:
		return ""

	decode_utf8 = np.vectorize(lambda v: v.decode("utf-8"))
	p = decode_utf8(p)
	s = ''.join(p)
	return s


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

	def __len__(self):
		return len(self.table)

	# -------------------------------------------------------------------------- #

	def lookup(self, value):
		try:
			return self.table.index(value)
		except ValueError:
			return UNK_ID

	def inverse_lookup(self, value):
		try:
			return self.table[value]
		except IndexError:
			return UNK

	def ids_to_string(self, line, output_as_array=False):
		d = [self.inverse_lookup(i) for i in line]
		if output_as_array:
			return d
		else:
			return ' '.join(d)

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

	def ids_to_english(self, line):
		line = self.ids_to_string(line)
		line = detokenize_english(line)
		return line


	def prediction_value_to_string(self, v, output_as_array=False):
		"""Rough 'n' ready get me the hell outta here fn. 
		Tries its best to deal with the mess of datatypes that end up coming out"""

		if isinstance(v, np.int64):
			s = self.inverse_lookup(v)
		elif isinstance(v, np.ndarray):
			if v.dtype == np.int64:
				s = self.ids_to_string(v, output_as_array)
			elif v.dtype == object:
				s = bytes_to_string(v)
			else:
				raise ValueError()
		else:
			raise ValueError()

		return s



	def save(self, args):
		with tf.gfile.GFile(args["vocab_path"], 'w') as out_file:
			for i in self.table:
				out_file.write(i + "\n")



	# --------------------------------------------------------------------------
	# Make me a vocab!
	# --------------------------------------------------------------------------
	

	@classmethod
	def load(cls, path, size):

		tokens = list()

		with tf.gfile.GFile(path) as file:
			for line in file.readlines():
				tokens.append(line.replace("\n", ""))
				
				if len(tokens) == size:
					break

		assert len(tokens) == len(set(tokens)), f"Duplicate lines in {path}"

		return Vocab(tokens)


	@classmethod
	def load_from_args(cls, args):
		return Vocab.load(args["vocab_path"], args["vocab_size"])



	@classmethod
	def build(cls, args, gqa_to_tokens):
		hits = Counter()

		def add(tokens:List[str]):
			for token in tokens:
				if token not in ["", " ", "\n"]:
					hits[token] += 1

		for i in tqdm(read_gqa(args), total=args["limit"]):
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




	