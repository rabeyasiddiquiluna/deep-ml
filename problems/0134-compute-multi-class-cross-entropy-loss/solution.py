import numpy as np

def compute_cross_entropy_loss(predicted_probs: np.ndarray, true_labels: np.ndarray, epsilon = 1e-15) -> float:

    predicted_probs = np.clip(predicted_probs, epsilon, 1 - epsilon)

    log_prob = - np.log(predicted_probs)
    loss = np.sum(log_prob * true_labels, axis = 1)

    return float(np.mean(loss))


    # Your code here
    pass