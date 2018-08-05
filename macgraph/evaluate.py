
from .args import get_args
from .estimator import get_estimator
from .input import gen_input_fn

if __name__ == "__main__":

	args = get_args()
	estimator = get_estimator(args)

	estimator.evaluate(input_fn=gen_input_fn(args, "eval"))