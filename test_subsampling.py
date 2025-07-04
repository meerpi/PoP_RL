import numpy as np
from final_env import PoPEnv

def test_env():
    print("Initializing environment...")
    env = PoPEnv(visual=False)
    env.reset()
    
    print("Populating episodic memory buffer...")
    # Force episodic memory to be large enough to trigger subsampling
    # We set ep_mem_size to 1500 and fill the buffer with random data
    env.ep_mem_size = 1500
    env.ep_mem_ptr = 1500
    env.ep_mem_buf[:1500] = np.random.rand(1500, 11).astype(np.float32)
    
    print("Testing compute_episodic_reward with subsampling...")
    dummy_z = np.random.rand(11).astype(np.float32)
    
    # This should trigger the new subsampling logic
    reward = env.compute_episodic_reward(dummy_z, sample_size=1000)
    print(f"Computed reward: {reward}")
    
    print("Testing standard environment steps...")
    for i in range(10):
        # sample random action
        action = env.action_space.sample()
        obs, reward_step, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            env.reset()
            
    print("All tests passed successfully.")

if __name__ == "__main__":
    test_env()
