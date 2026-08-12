def binary_cross_entropy(y_true: list[float], y_pred: list[float], epsilon: float = 1e-15) -> float:
	import numpy as np
	"""
	Compute binary cross-entropy loss.
	
	Args:
		y_true: True binary labels (0 or 1)
		y_pred: Predicted probabilities (between 0 and 1)
		epsilon: Small value for numerical stability
	
	Returns:
		Mean binary cross-entropy loss
	"""
	# Your code here
	y_true = np.array(y_true)
	y_pred = np.array(y_pred)
	p = np.clip(y_pred, epsilon, 1-epsilon)
	bce = -np.mean(y_true *np.log(p) + (1-y_true)*np.log(1-p))


	return float(bce)
	pass