# Checking the logic for the episodic memory:
# In final_env.py:
#         self.ep_mem_buf[self.ep_mem_ptr] = z_t
#         self.ep_mem_ptr = (self.ep_mem_ptr + 1) % self.ep_mem_buf.shape[0]
#         if self.ep_mem_size < self.ep_mem_buf.shape[0]:
#             self.ep_mem_size += 1

# And in compute_episodic_reward:
#         if self.ep_mem_size == 0:
#             return 1.0

# In the paper, the reward is computed BEFORE adding the memory?
# Yes. Algorithm 1, line 14: compute r_episodic.
# Line 15: add z_i to episodic memory M.
# In Sahil's code, it's computed before adding:
#     obs = self._get_obs()
#     episodic_z = self.get_mem_step()
#     r_episodic = self.compute_episodic_reward(episodic_z)
#     self.add_to_episodic_memory(episodic_z)
# This is correct.

# What about the dimension of z_t?
# It's an array of length 2 [gx/250.0, gy/100.0]
# Diff squared shape:
# sq_dist = (diff ** 2).sum(axis=1) (N)
# This is Euclidean distance squared. 
# The paper says: "squared Euclidean distance". So this is correct.

# What about eps, xi, c in compute_episodic_reward?
# def compute_episodic_reward(self, z_t, k=10, eps=0.0001, xi=0.008, c=0.001, s_max=8.0):
# Paper defaults: eps=0.001 (not 0.0001), xi=0.008, c=0.001. Wait, let me double check eps.
# The paper says eps = 10^-3 or 10^-4?
# Appendix A "we set pm = 10^-4 ... epsilon = 10^-4". So eps=0.0001 is correct.

# Wait, there is another bug in final_env.py step():
#         n = self.room_visit_counts.get(key, 0)
#         alpha_t = 1.0 / np.sqrt(n / 10.0 + 1)
#         alpha_t = float(np.clip(alpha_t * 5.0, 1.0, 5.0))
# Notice it increments room_visit_counts ONLY when changing rooms:
#         if self.kid_room != self._prev_kid_room:
#             n = self.room_visit_counts.get(key, 0)
#             self.room_visit_counts[key] = n + 1
#             self._prev_kid_room = self.kid_room
# Wait, this means if you spawn in a room, you get count=1.
# If you stay in the room for 2000 steps, alpha_t NEVER drops because it only updates when you changing rooms!
# But alpha_t is supposed to scale down based on the number of times you have visited the room/state.
# If alpha_t stays high, then as long as you're in the room, you get high rewards. 
# NO, wait.
# The episodic memory tracks every timestep. So r_episodic decays as you stay in the room.
# alpha_t is lifelong novelty.
# The paper says alpha_t is based on the count of the state across all episodes.
# "N_t(xi)" is the count. If a room is considered a state, it makes sense to count visits to the room, not frames spent in the room.

print("No other obvious NGU math bugs found.")
