
class RecordWriter(object):

	def __init__(self, args, mode):
		self.args = args
		self.mode = mode

	def __enter__(self, *vargs):
		self.file = tf.python_io.TFRecordWriter(self.args[f"{self.mode}_input_path"]) 
		return self

	def write(self, doc, record):
		self.file.write(record)

	def close(self):
		self.file.close()

	def __exit__(self, *vargs):
		self.file.close()
		self.file = None


class Partitioner(object):
	"""
	Write to this and it'll randomly write to the writer_dict
	writers.

	It is the callers responsibility to close the writer_dicts after
	this has been disposed of.

	Arguments:
		writer_dict: A dictionary of mode strings to something that accepts a write(doc, record) call
	"""

	def __init__(self, args, writer_dict):
		self.args = args
		self.written = 0
		self.answer_classes = Counter()
		self.answer_classes_types = Counter()
		self.writer_dict = writer_dict

	def __enter__(self, *vargs):
		pass


	def write(self, doc, record):
		r = random.random()

		if r < self.args["eval_holdback"]:
			mode = "eval"
		elif r < self.args["eval_holdback"] + self.args["predict_holdback"]:
			mode = "predict"
		else:
			mode = "train"

		key = (str(doc["answer"]), doc["question"]["type_string"])

		self.writer_dict[mode].write(doc, record)
		self.answer_classes[str(doc["answer"])] += 1
		self.answer_classes_types[key] += 1
		self.written += 1


	def __exit__(self, *vargs):
		pass

