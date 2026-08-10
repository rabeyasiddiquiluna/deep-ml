def activation_derivatives(x: float) -> dict[str, float]:
	"""
	Compute the derivatives of Sigmoid, Tanh, and ReLU at a given point x.
	
	Args:
		x: Input value
		
	Returns:
		Dictionary with keys 'sigmoid', 'tanh', 'relu' and their derivative values
	"""
	# Your code here
	import math
	sig = 1.0 / (1.0 + math.exp(-x))
	d_sigmoid = sig * (1- sig)
	d_tanh = 1.0 - math.tanh(x) ** 2
	d_relu = 1.0 if x > 0 else 0.0

	return {
        'sigmoid': d_sigmoid,
        'tanh': d_tanh,
        'relu': d_relu
    }
	pass