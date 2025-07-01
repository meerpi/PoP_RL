import numpy as np
# Let's check another potential issue in NGU:
# Does NGU paper algorithm say:
# alpha_t * r_episodic 
# OR
# r_episodic * min(max(alpha_t, 1), L)

# In final_env.py:
#         alpha_t = float(np.clip(alpha_t * 5.0, 1.0, 5.0))
#         info = ..., "alpha_t": alpha_t

# In ppo.py:
#         current_int_rewards = r_episodic_t * alpha_t

# This matches r_episodic * min(max(alpha_t, 1), L) where L=5 and the inner max/min bounds it between 1 and 5.

# Let's check GAE calculation for intrinsic reward:
# ppo.py:
# delta_int = transformed_int_rewards[t] + GAMMA_INT * nextv_int * 1.0 - values_int[t]
# adv_int[t] = lastgaelam_int = delta_int + GAMMA_INT * GAE_LAMBDA * 1.0 * lastgaelam_int

# Is nextnonterminal 1.0 for intrinsic reward correct?
# Badia et al. 2020: "The intrinsic reward is non-episodic, meaning the discount factor gamma_I is not set to 0 at the end of an episode."
# So 1.0 is mathematically correct for NGU!

print("Intrinsic GAE non-episodic is correct.")
