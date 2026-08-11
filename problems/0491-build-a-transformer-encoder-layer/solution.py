import numpy as np

def transformer_encoder_layer(X: np.ndarray, weights: dict, num_heads: int, eps: float = 1e-5) -> np.ndarray:
    """
    Forward pass of a single Transformer Encoder Layer.

    Args:
        X: Input tensor of shape (batch_size, seq_len, d_model)
        weights: Dictionary containing all weight matrices and normalization parameters
        num_heads: Number of attention heads
        eps: Epsilon for layer normalization

    Returns:
        Output tensor of shape (batch_size, seq_len, d_model)
    """
    # Your code here

    batch_size, seq_len , d_model = X.shape
    W_q = weights['W_q']
    W_k = weights['W_k']
    W_v = weights['W_v']
    W_o = weights['W_o']
    W1 = weights['W1']
    W2 = weights['W2']
    b1 = weights['b1']
    b2 = weights['b2']
    gamma1 = weights['gamma1']
    gamma2 = weights['gamma2']
    beta1 = weights['beta1']
    beta2 = weights['beta2']

    d_k = d_model // num_heads
    Q = X @W_q
    K = X @W_k
    V = X @W_v

    Q = Q.reshape(batch_size, seq_len,num_heads,d_k).transpose(0,2,1,3)
    K = K.reshape(batch_size, seq_len,num_heads,d_k).transpose(0,2,1,3)
    V = V.reshape(batch_size, seq_len,num_heads,d_k).transpose(0,2,1,3)

    scores = Q @K.transpose(0,1,3,2)/np.sqrt(d_k)
    max_scores = np.max(scores, axis = -1, keepdims = True)
    attn_weights = np.exp(scores - max_scores)
    attention_weights = attn_weights/ np.sum(attn_weights, axis = -1, keepdims = True)

    attention_output = attention_weights @V

    attention_output_concat = attention_output.transpose(0,2,1,3).reshape(batch_size, seq_len , d_model )

    output_projection = attention_output_concat @ W_o

    # step 2 : Add & Norm(first)

    X1 = X + output_projection
    X_mean = np.mean(X1, axis = -1, keepdims = True)
    X_var = np.var(X1, axis = -1, keepdims = True)
    X_norm = gamma1 * (X1 - X_mean) / np.sqrt(X_var+ eps) + beta1

    #step 3 : feedforward nwtwork
    hidden_network = X_norm @W1 + b1
    hidden_network = np.maximum(0, hidden_network)
    ff_output = hidden_network @W2+ b2

    #step4 : Add & norm
    X2 = X_norm + ff_output
    mean2= np.mean(X2, axis = -1, keepdims = True)
    var2 = np.var(X2, axis = -1, keepdims = True)
    output = gamma2 * (X2 - mean2)/np.sqrt(var2+eps) + beta2

    return output



    
    pass