def softsign(x: float) -> float:

	import numpy as np
	"""
	Implements the Softsign activation function.

	Args:
		x (float): Input value

	Returns:
		float: The Softsign of the input
	"""
	# Your code here
	x = np.asarray (x, dtype = float)
	result = x / (1 + np.abs(x))
	return round(result, 4)