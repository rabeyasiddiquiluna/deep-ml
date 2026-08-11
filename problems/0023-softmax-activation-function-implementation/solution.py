import math

def softmax(scores: list[float]) -> list[float]:
    # subtract max for numerical stability(prevents overflow)
    max_scores = max(scores)
    exp_scores = [math.exp(s - max_scores) for s in scores]
    sum_exp = sum(exp_scores)
    return [e / sum_exp for e in exp_scores]

