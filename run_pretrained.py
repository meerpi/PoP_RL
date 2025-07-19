import os
import sys
import time

root_dir = os.path.abspath(os.path.dirname(__file__))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.monitor import Monitor
from POP_env import POPEnv

def main():
    model_path = os.path.abspath(os.path.join(root_dir, "models", "ppo_final"))
    
    mode = "1"
    if len(sys.argv) >= 2:
        if sys.argv[1] in ["1", "2"]:
            mode = sys.argv[1]
        else:
            print("Invalid argument. Use: python run_pretrained.py [1|2]")
            print("  1: GUI Window (Default)")
            print("  2: Headless (No window)")
            sys.exit(1)
            
    print(f"Loading pretrained model from: {model_path}.zip")
    if not os.path.exists(f"{model_path}.zip"):
        print(f"Error: Pretrained model file not found at {model_path}.zip")
        sys.exit(1)

    print(f"Initializing POPEnv in {'GUI Window' if mode == '1' else 'Headless'} mode...")
    
    if mode == "1":
        if "SDL_VIDEODRIVER" in os.environ:
            del os.environ["SDL_VIDEODRIVER"]
    else:
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    def make_env():
        env = POPEnv()
        env = Monitor(env)
        return env

    env = DummyVecEnv([make_env])
    env = VecTransposeImage(env)

    try:
        device = "cuda" if os.path.exists("/dev/nvidia0") or os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
        print(f"Loading model on device: {device}")
        model = PPO.load(model_path, env=env, device=device)
        print("Model loaded successfully!")
        
        print("\nStarting evaluation! Press Ctrl+C to stop.\n")
        obs = env.reset()
        
        while True:
            action, _states = model.predict(obs, deterministic=True)
            
            obs, rewards, dones, infos = env.step(action)
            
            info = infos[0]
            print(
                f"Step: {info['step']:<5} | "
                f"HP: {info['hp']:<2} | "
                f"Sword: {info['has_sword']:<2} | "
                f"Episode Reward: {info['total_reward']:.2f}",
                end="\r"
            )
            
            if dones[0]:
                print(f"\n--- Episode Finished! Auto-resetting environment... ---\n")
                
            if mode == "1":
                time.sleep(0.02)

    except KeyboardInterrupt:
        print("\n\nEvaluation interrupted by user.")
    finally:
        print("Closing environment...")
        env.close()

if __name__ == "__main__":
    main()
