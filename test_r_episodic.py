import numpy as np

class EpisodicMemory:
    def __init__(self):
        self.ep_mem_buf  = np.zeros((30_000, 11), dtype=np.float32)
        self.ep_mem_size = 0
        self.ep_mem_ptr  = 0
        self.ep_d2m      = 1.0

    def compute_episodic_reward(self, z_t, k=10, eps=0.0001, xi=0.008, c=0.001, s_max=8.0):
        if self.ep_mem_size == 0:
            return 1.0 / np.sqrt(c)
        valid = self.ep_mem_buf[:self.ep_mem_size]
        diff = valid - z_t
        sq_dist = (diff ** 2).sum(axis=1)
        k_actual = min(k, len(valid))
        nn_idx = np.argpartition(sq_dist, k_actual - 1)[:k_actual]
        dk = sq_dist[nn_idx]
        for d in dk:
            self.ep_d2m = self.ep_d2m * 0.99 + d * 0.01
        dn = dk / max(self.ep_d2m, 1e-8)
        dn = np.maximum(dn - xi, 0.0)
        Kv = eps / (dn + eps)
        s = np.sqrt(Kv.sum() + c)
        if s > s_max:
            return 0.0
        return 1.0 / s

    def add_to_episodic_memory(self, z_t):
        self.ep_mem_buf[self.ep_mem_ptr] = z_t
        self.ep_mem_ptr = (self.ep_mem_ptr + 1) % self.ep_mem_buf.shape[0]
        if self.ep_mem_size < self.ep_mem_buf.shape[0]:
            self.ep_mem_size += 1

mem = EpisodicMemory()
rewards = []
# Simulate stuck in same state
for i in range(100):
    z_t = np.zeros(11, dtype=np.float32)
    r = mem.compute_episodic_reward(z_t)
    mem.add_to_episodic_memory(z_t)
    rewards.append(r)

print("Stuck:")
print(f"Max: {np.max(rewards):.2f}, Min: {np.min(rewards):.2f}, Mean: {np.mean(rewards):.2f}")

mem2 = EpisodicMemory()
rewards2 = []
gx = 0.0
for i in range(100):
    gx += 0.012
    z_t = np.zeros(11, dtype=np.float32)
    z_t[0] = gx
    r = mem2.compute_episodic_reward(z_t)
    mem2.add_to_episodic_memory(z_t)
    rewards2.append(r)

print("Linear movement:")
print(f"Max: {np.max(rewards2):.2f}, Min: {np.min(rewards2):.2f}, Mean: {np.mean(rewards2):.2f}")
