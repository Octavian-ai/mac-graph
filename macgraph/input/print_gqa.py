
import tableprint
from collections import Counter

from .args import *
from .util import *


if __name__ == "__main__":

	args = get_args()

	output_classes = Counter()
	question_types = Counter()


	with tableprint.TableContext(headers=["Type", "Question", "Answer"], width=[40,50,15]) as t:
		for i in read_gqa(args):
			output_classes[i["answer"]] += 1
			question_types[i["question"]["type_string"]] += 1
			t([
				i["question"]["type_string"],
				i["question"]["english"], 
				i["answer"]
			])

	def second(v):
		return v[1]

	tableprint.table(headers=["Answer", "Count"],   width=[20,5], data=sorted(output_classes.items(), key=second))
	tableprint.table(headers=["Question", "Count"], width=[20,5], data=sorted(question_types.items(), key=second))


