
from collections import Counter
import numpy as np
import string
import re
import tensorflow as tf
import sys
from tqdm import tqdm
import os.path
import zipfile
import urllib.request
import pathlib

from .util import *
from ..args import *


import logging
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------

UNK = "<unk>"
SOS = "<sos>"
EOS = "<eos>"
SPACE = "<space>"
SPECIAL_TOKENS = [UNK, SOS, EOS, SPACE]

UNK_ID = SPECIAL_TOKENS.index(UNK)
SOS_ID = SPECIAL_TOKENS.index(SOS)
EOS_ID = SPECIAL_TOKENS.index(EOS)

ENGLISH_PUNCTUATION = '!"#$%&()*+,-./:;=?@[\\]^_`{|}~'

# --------------------------------------------------------------------------


def build_vocab(args):

	hits = Counter()

	def add_line(line):
		line = line.replace("\n", "")

		for word in line.split(' '):
			if word != "" and word != " ":
				hits[word] += 1

	for i in read_gqa(args):
		add_line(i["question"]["english"])

	tokens = list()
	tokens.extend(SPECIAL_TOKENS)

	for i in string.ascii_lowercase:
		tokens.append("<"+i+">")
		tokens.append("<"+i.upper()+">")

	for i, c in hits.most_common(args["vocab_size"]):
		if len(tokens) == args["vocab_size"]:
			break

		if i not in tokens:
			tokens.append(i)

	assert len(tokens) <= args["vocab_size"]

	with tf.gfile.GFile(args["vocab_path"], 'w') as out_file:
		for i in tokens:
			out_file.write(i + "\n")

	return tokens



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

def lookup_vocab(token, vocab):
	try:
		return vocab.index(token)
	except ValueError:
		return UNK_ID


def string_to_tokens(line, vocab):
	s = line.split(' ')

	r = []
	for i in s:
		r.append(lookup_vocab(i, vocab))

	return r



def pretokenize_general(text):
	text = re.sub(r'\s*$', '', text)
	text = text.replace(" ", f" {SPACE} ")
	return text


def pretokenize_json(value):

	def to_str(value):
		if value == True:
			return 'y'
		elif value == False:
			return 'f'
		elif isinstance(value, str):
			return value
		elif isinstance(value, list):
			return ' '.join(value)
		else:
			return str(value)

	return pretokenize_general(to_str(value))


def pretokenize_english(text):

	text = pretokenize_general(text)

	for p in ENGLISH_PUNCTUATION:
		text = text.replace(p, f" {p} ")

	return text



