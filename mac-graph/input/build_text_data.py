
import yaml
import sys
import tensorflow as tf
import random
from collections import Counter
from tqdm import tqdm
import os.path
import zipfile
import urllib.request
import pathlib

from .util import *
from ..args import *



def build_vocab(args):

	hits = Counter()

	def add_lines(lines):
		for line in lines:
			line = line.replace("\n", "")

			for word in line.split(' '):
				if word != "" and word != " ":
					hits[word] += 1

	for i in ["all"]:
		for j in ["src", "tgt"]:
			with tf.gfile.GFile(args[f"{i}_{j}_path"]) as in_file:
				add_lines(in_file.readlines())

	tokens = list()
	tokens.extend(special_tokens)

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




