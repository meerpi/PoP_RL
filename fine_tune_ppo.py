import gymnasium
import os
import sys
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecTransposeImage
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from POP_env import POPEnv

root_dir = os.path.abspath(os.path.dirname(__file__))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

def make_env(rank, mode, seed=0):
    def _init():
        if mode == "1":
            if rank == 0:
                if "SDL_VIDEODRIVER" in os.environ:
                    del os.environ["SDL_VIDEODRIVER"]
            else:
                os.environ["SDL_VIDEODRIVER"] = "dummy"
        elif mode == "2":
            if "SDL_VIDEODRIVER" in os.environ:
                del os.environ["SDL_VIDEODRIVER"]
        elif mode == "3":
            os.environ["SDL_VIDEODRIVER"] = "dummy"

        env = POPEnv()
        env.reset(seed=seed + rank)
        env = Monitor(env)
        return env
    
    set_random_seed(seed)
    return _init

def train(mode):
    num_cpu = 12
    bc_model_path = os.path.abspath(os.path.join(root_dir, "models", "ppo_final"))
    
    print("=" * 60)
    print("RL Fine-Tuning of BC-Pretrained Model using PPO")
    print("=" * 60)
    print(f"Mode selected: {mode}")
    print(f"Starting training on {num_cpu} environments...")
    
    if not os.path.exists(f"{bc_model_path}.zip"):
        print(f"Error: Behavioral Cloning model not found at '{bc_model_path}.zip'!")
        print("Please run 'python train_bc.py' first to pretrain the policy.")
        sys.exit(1)

    env = SubprocVecEnv([make_env(i, mode) for i in range(num_cpu)])
    env = VecTransposeImage(env)

    try:
        print(f"\nLoading imitation learning weights from '{bc_model_path}.zip'...")
        
        custom_objects = {
            "n_steps": 4096,
            "batch_size": 1024,
            "n_epochs": 10,
            "tensorboard_log": "logs/",
            "learning_rate": 2.5e-4
        }
        
        model = PPO.load(
            bc_model_path,
            env=env,
            custom_objects=custom_objects,
            device="cuda"
        )
        print("✓ Pretrained model loaded and bound to new environment vector successfully!")
        
        print("\nStarting PPO reinforcement learning fine-tuning...")
        print("Access TensorBoard logs under logs/ to view learning progress.")
        
        model.learn(total_timesteps=8000000, progress_bar=True)
        
        models_dir = os.path.abspath(os.path.join(root_dir, "models"))
        os.makedirs(models_dir, exist_ok=True)
        save_path = os.path.join(models_dir, "ppo_fine_tuned")
        print(f"\nTraining completed. Saving fine-tuned model to '{save_path}.zip'...")
        model.save(save_path)
        print(f"✓ Save complete to {save_path}.zip!")
        
    except KeyboardInterrupt:
        models_dir = os.path.abspath(os.path.join(root_dir, "models"))
        os.makedirs(models_dir, exist_ok=True)
        interrupted_path = os.path.join(models_dir, "ppo_fine_tuned_interrupted")
        print(f"\nFine-tuning interrupted by user. Saving current checkpoint to '{interrupted_path}.zip'...")
        model.save(interrupted_path)
        print("✓ Interrupted checkpoint saved successfully.")
    finally:
        env.close()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python fine_tune_ppo.py [1|2|3]")
        print("  1: One window visible (ideal for visual confirmation)")
        print("  2: All windows visible")
        print("  3: Headless mode (fastest)")
    else:
        user_mode = sys.argv[1]
        train(user_mode)
