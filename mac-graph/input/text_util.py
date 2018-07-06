
from collections import Counter
import numpy as np
import string
import re
import tensorflow as tf

import logging
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------

UNK = "<unk>"
SOS = "<sos>"
EOS = "<eos>"
SPACE = "<space>"
special_tokens = [UNK, SOS, EOS, SPACE]

UNK_ID = special_tokens.index(UNK)
SOS_ID = special_tokens.index(SOS)
EOS_ID = special_tokens.index(EOS)

ENGLISH_PUNCTUATION = '!"#$%&()*+,-./:;=?@[\\]^_`{|}~'

# --------------------------------------------------------------------------

def load_vocab(args):
	tokens = list()

	with tf.gfile.GFile(args["vocab_path"]) as file:
		for line in file.readlines():
			tokens.append(line.replace("\n", ""))
			
			if len(tokens) == args["vocab_size"]:
				return tokens


	return tokens

# --------------------------------------------------------------------------


def expand_unknown_vocab(line, vocab):
	ts = set(line.split(' '))
	unknowns = ts
	unknowns -= set(vocab)
	unknowns -= set([''])

	for t in unknowns:
		spaced = ''.join([f"<{c}> " for c in t])
		line = line.replace(t, spaced)

	return line

def string_to_tokens(line, vocab):
	s = line.split(' ')

	r = []
	for i in s:
		try:
			r.append(vocab.index(i))
		except ValueError:
			r.append(UNK_ID)

	return r



def pretokenize_general(text):
	text = re.sub(r'\s*$', '', text)
	text = text.replace(" ", f" {SPACE} ")
	return text





def pretokenize_english(text):

	text = pretokenize_general(text)

	for p in ENGLISH_PUNCTUATION:
		text = text.replace(p, f" {p} ")

	# From Keras Tokenizer
	# filters = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
	# split = ' '

	# translate_map = str.maketrans(filters, split * len(filters))
	# text = text.translate(translate_map)

	return text



