import numpy as np
def train_neuron(features: np.ndarray, labels: np.ndarray, initial_weights: np.ndarray, initial_bias: float, learning_rate: float, epochs: int) -> (np.ndarray, float, list[float]):
	# Your code here


	X = np.array(features, dtype = float)
	Y = np.array(labels, dtype = float)
	W = np.array(initial_weights, dtype = float)
	b = float(initial_bias)
	lr = float(learning_rate)

	N = X.shape[0]
	mse_values = []

	for _ in range(epochs):
		z = X @W + b
		prob = 1/ ( 1 + np.exp(- z))
		err = prob - Y
		mse_values.append(round(float(np.mean( err ** 2)),4))

		dz = 2/N * prob * err * (1 - prob)
		dw = X.T @ dz
		db = np.sum(dz)

		W  -= lr * dw
		b -= lr * db




	return np.round(W, 4).tolist(), round(b,4), mse_values