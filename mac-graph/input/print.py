
import tableprint
from collections import Counter

from .args import *
from .util import *


if __name__ == "__main__":

	args = get_args()

	answer_classes = Counter()


	with tableprint.TableContext(headers=["Type", "Question", "Answer"], width=[30,40,15]) as t:
		for i in read_gqa(args):
			answer_classes[i["answer"]] += 1
			t([
				i["question"]["type_string"],
				i["question"]["english"], 
				i["answer"]
			])


	tableprint.table(headers=["Type", "Count"], data=answer_classes.items())

