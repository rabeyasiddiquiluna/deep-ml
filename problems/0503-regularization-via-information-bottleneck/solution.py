import numpy as np

def information_bottleneck_loss(task_loss: float, mu: np.ndarray, log_var: np.ndarray, beta: float) -> tuple:
    """
    Compute the Information Bottleneck regularized loss.
    
    Args:
        task_loss: Precomputed task loss (scalar)
        mu: Encoder means, shape (batch_size, latent_dim)
        log_var: Encoder log-variances, shape (batch_size, latent_dim)
        beta: Trade-off parameter for compression regularization
    
    Returns:
        Tuple of (total_loss, mean_kl_divergence), both rounded to 4 decimal places
    """
    pass

    kl_loss = - .5 * np.sum(1+log_var-np.exp(log_var)-mu**2, axis = -1)
    mean_kl = np.mean(kl_loss)
    total_loss = task_loss + beta * mean_kl

    return (round(total_loss,4), round(mean_kl, 4))

