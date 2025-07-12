import os
import sys
import numpy as np

# Ensure root directory is in sys.path
root_dir = os.path.abspath(os.path.dirname(__file__))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# Resolve expert model path absolutely
expert_model_path = os.path.abspath(os.path.join(root_dir, "models", "trained_ppo"))

if not os.path.exists(f"{expert_model_path}.zip"):
    print(f"Error: Expert model not found at '{expert_model_path}.zip'!")
    print("Please make sure your trained model is named 'trained_ppo.zip' inside the 'models/' directory.")
    sys.exit(1)

# Default harvesting options
mode = "2"  # Headless (fastest collection)
num_episodes = 10

# Parse command line options
if len(sys.argv) >= 2:
    if sys.argv[1] in ["1", "2"]:
        mode = sys.argv[1]
    else:
        print("Usage: python harvest_demonstrations.py [1|2] [num_episodes]")
        print("  1: GUI Mode (Slow, watch harvesting)")
        print("  2: Headless Mode (Fastest, default)")
        sys.exit(1)
        
if len(sys.argv) >= 3:
    try:
        num_episodes = int(sys.argv[2])
    except ValueError:
        print("Error: num_episodes must be an integer.")
        sys.exit(1)

# Set environment variables for POPEnv video driver before imports
if mode == "1":
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
else:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.monitor import Monitor
from POP_env import POPEnv

def make_env():
    env = POPEnv()
    env = Monitor(env)
    return env

def main():
    print("=" * 60)
    print("Expert Demonstration Harvesting Script")
    print("=" * 60)
    print(f"Loading expert model: {expert_model_path}.zip")
    print(f"Harvesting Mode: {'GUI' if mode == '1' else 'Headless'}")
    print(f"Target Episodes: {num_episodes}")
    print("=" * 60 + "\n")

    # Initialize env framework
    env = DummyVecEnv([make_env])
    env = VecTransposeImage(env)

    try:
        # Load the expert model
        device = "cuda" if os.path.exists("/dev/nvidia0") or os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
        print(f"Loading model on device: {device}...")
        model = PPO.load(expert_model_path, env=env, device=device)
        print("✓ Expert model loaded successfully!")

        pixels_list = []
        state_list = []
        action_list = []

        total_steps = 0
        
        for ep in range(1, num_episodes + 1):
            obs = env.reset()
            done = False
            ep_steps = 0
            ep_reward = 0.0
            
            # Temporary storage for the current episode
            ep_pixels = []
            ep_state = []
            ep_actions = []

            while not done:
                # Reconstruct original channel-last observation shape from the VecTransposeImage output
                # obs['pixels'] is shape (1, 1, 84, 84), we transpose it back to original (84, 84, 1)
                pixels_harvested = np.transpose(obs['pixels'].squeeze(0), (1, 2, 0))
                state_harvested = obs['state'].squeeze(0)  # Shape (4,)

                # Predict deterministic action using the expert model
                action, _states = model.predict(obs, deterministic=True)
                
                # Step environment
                next_obs, rewards, dones, infos = env.step(action)
                
                # Record components
                ep_pixels.append(pixels_harvested)
                ep_state.append(state_harvested)
                ep_actions.append(action[0])

                ep_steps += 1
                total_steps += 1
                ep_reward += rewards[0]
                done = dones[0]

                print(
                    f"Episode {ep}/{num_episodes} | "
                    f"Step: {ep_steps:<4} | "
                    f"Total Harvested: {total_steps:<5} | "
                    f"HP: {infos[0]['hp']:<2} | "
                    f"Sword: {infos[0]['has_sword']}",
                    end="\r"
                )

            # Append episode data to main dataset lists
            pixels_list.extend(ep_pixels)
            state_list.extend(ep_state)
            action_list.extend(ep_actions)

            print(
                f"\n✓ Episode {ep} completed: "
                f"Steps = {ep_steps:<4} | "
                f"Reward = {ep_reward:.2f} | "
                f"Sword = {infos[0]['has_sword']}\n"
            )

        # Save harvested dataset
        data_dir = os.path.abspath(os.path.join(root_dir, "data"))
        os.makedirs(data_dir, exist_ok=True)
        target_file = os.path.join(data_dir, "demonstrations.npz")

        print(f"Saving {total_steps} harvested transitions to '{target_file}'...")
        np.savez_compressed(
            target_file,
            pixels=np.array(pixels_list, dtype=np.uint8),
            state=np.array(state_list, dtype=np.float32),
            actions=np.array(action_list, dtype=np.int64)
        )
        print("✓ Harvesting complete and data successfully saved!")

    except KeyboardInterrupt:
        print("\n\nHarvesting interrupted by user. No data saved.")
    finally:
        env.close()

if __name__ == "__main__":
    main()
