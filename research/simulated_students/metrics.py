"""Small, dependency-free statistics for comparing simulated and real students."""
import math
import random


def _ranks(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        average = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = average
        i = j + 1
    return ranks


def pearson(x, y):
    n = len(x)
    if n < 2:
        return float('nan')
    mean_x, mean_y = sum(x) / n, sum(y) / n
    sxy = sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y))
    sxx = sum((a - mean_x) ** 2 for a in x)
    syy = sum((b - mean_y) ** 2 for b in y)
    if sxx == 0 or syy == 0:
        return float('nan')
    return sxy / math.sqrt(sxx * syy)


def spearman(x, y):
    return pearson(_ranks(list(x)), _ranks(list(y)))


def mae(x, y):
    return sum(abs(a - b) for a, b in zip(x, y)) / len(x)


def bootstrap_ci(x, y, statistic, rounds=2000, seed=0, level=0.95):
    """Percentile interval for statistic(x, y), resampling paired items."""
    rng = random.Random(seed)
    n = len(x)
    values = []
    for _ in range(rounds):
        picks = [rng.randrange(n) for _ in range(n)]
        value = statistic([x[i] for i in picks], [y[i] for i in picks])
        if not math.isnan(value):
            values.append(value)
    values.sort()
    if not values:
        return float('nan'), float('nan')
    tail = (1 - level) / 2
    return values[int(tail * (len(values) - 1))], values[int((1 - tail) * (len(values) - 1))]


def jsd(p, q):
    """Jensen-Shannon divergence (base 2, between 0 and 1) of two distributions given as dicts."""
    keys = set(p) | set(q)
    m = {k: (p.get(k, 0.0) + q.get(k, 0.0)) / 2 for k in keys}

    def kl(a):
        return sum(a.get(k, 0.0) * math.log2(a.get(k, 0.0) / m[k]) for k in keys if a.get(k, 0.0) > 0)

    return (kl(p) + kl(q)) / 2


def top_distractor(shares, correct_id):
    """The most chosen wrong option, or None if no wrong option was chosen."""
    wrong = {k: v for k, v in shares.items() if k != str(correct_id) and v > 0}
    if not wrong:
        return None
    return max(sorted(wrong), key=lambda k: wrong[k])


def wasserstein(a, b):
    """Earth mover's distance between two histograms {value: count} (values as strings or numbers):
    how far, on average, scores must move to turn one distribution into the other."""
    total_a, total_b = sum(a.values()), sum(b.values())
    if not total_a or not total_b:
        return None
    points = sorted({float(k) for k in a} | {float(k) for k in b})
    pa = {float(k): v / total_a for k, v in a.items()}
    pb = {float(k): v / total_b for k, v in b.items()}
    distance = cdf_a = cdf_b = 0.0
    for left, right in zip(points, points[1:]):
        cdf_a += pa.get(left, 0.0)
        cdf_b += pb.get(left, 0.0)
        distance += abs(cdf_a - cdf_b) * (right - left)
    return distance
