import numpy as np

def compute_episodic_reward(valid, z_t, k=10, eps=0.0001, xi=0.008, c=0.001, ep_d2m=0.01):
    diff = valid - z_t
    sq_dist = (diff ** 2).sum(axis=1)
    k_actual = min(k, len(valid))
    
    # Bug 3 check: nn_idx is missing the sort?
    nn_idx = np.argpartition(sq_dist, k_actual - 1)[:k_actual]
    dk = sq_dist[nn_idx]
    
    dk_sorted = np.sort(dk)
    ep_d2m = ep_d2m * 0.99 + dk_sorted[-1] * 0.01
    
    dn = dk / max(ep_d2m, 1e-8)
    dn = np.maximum(dn - xi, 0.0)
    Kv = eps / (dn + eps)
    s = np.sqrt(Kv.sum() + c)
    return 1.0 / s, ep_d2m

z_t = np.array([1.0, 1.0])
valid = np.array([[1.0, 1.0]] * 15)
print(compute_episodic_reward(valid, z_t))

valid = np.random.randn(20, 2)
z_t = np.array([10.0, 10.0]) # far away
print(compute_episodic_reward(valid, z_t))

