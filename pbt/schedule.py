

def decay_schedule(start_val=40, end_val=10, decay_period=30):
	return lambda epoch: round(end_val + start_val * max(decay_period-epoch,0)/decay_period)
