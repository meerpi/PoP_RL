import numpy as np

# NGU algorithm 1:
# Equation 1: r_episodic = 1 / sqrt( sum(eps / (d^2 / d^2_m + eps)) + c )
# Algorithm 1 has:
# r_episodic = 1/s

# Equation 2: alpha_i = 1 + c1 / sqrt(n(xi_i))
# However NGU paper Appendix:
# Let L = 5
# b_i = alpha_i * r_episodic

# Bug 4: NGU algorithm 1 line 12:
# "Compute the modulation factor: alpha = 1 + (L - 1) / sqrt(n(x_k)) "  (or similar)
# Let's check NGU paper Algorithm 1 (pg 14 in NGU paper):
# Line 10: "α_i ← 1 + 0.1 / √N_t(xi_i)"  ??? 
# Wait, let me check the exact paper formula for alpha in NGU. 
# Equation 1 from NGU:
# rE_t = 1 / sqrt( sum_j K(x, x_j) + c )
# rI_t = rE_t * min(max(alpha_t, 1), L)
# alpha_t = 1 + alpha_0 / sqrt(N(x_t))

# Let's check what Sahil's code is doing:
# n = self.room_visit_counts.get(key, 0)
# alpha_t = 1.0 / np.sqrt(n / 10.0 + 1)
# alpha_t = float(np.clip(alpha_t * 5.0, 1.0, 5.0))

# This looks completely different!
# Sahil's alpha goes from 5.0 down to 1.0. 
# NGU alpha = 1 + c / sqrt(N). It goes from (1+c) down to 1.0.

print("Checking...")
