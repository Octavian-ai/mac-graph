
import tableprint
from collections import Counter
from tqdm import tqdm

from .args import *
from .util import *


if __name__ == "__main__":

	def extend(parser):
		parser.add_argument("--print-rows",action='store_true')

	args = get_args(extend)

	output_classes = Counter()
	question_types = Counter()
	questions = Counter()

	try:
		with tableprint.TableContext(headers=["Type", "Question", "Answer"], width=[40,50,15]) as t:
			for i in read_gqa(args):
				output_classes[i["answer"]] += 1
				question_types[i["question"]["type_string"]] += 1
				questions[i["question"]["english"]] += 1

				if args["print_rows"]:
					t([
						i["question"]["type_string"],
						i["question"]["english"], 
						i["answer"]
					])

	except KeyboardInterrupt:
		print()
		pass
		# we want to print the final results!

	def second(v):
		return v[1]

	tableprint.table(headers=["Answer", "Count"],   width=[20,5], data=sorted(output_classes.items(), key=second))
	tableprint.table(headers=["Question", "Count"], width=[50,5], data=sorted(questions.items(), key=second))
	tableprint.table(headers=["Question type", "Count"], width=[20,5], data=sorted(question_types.items(), key=second))
	



