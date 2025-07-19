import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# Ensure root directory is in sys.path
root_dir = os.path.abspath(os.path.dirname(__file__))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.monitor import Monitor
from POP_env import POPEnv

# PyTorch Dataset for loading demonstrations
class DemoDataset(Dataset):
    def __init__(self, pixels, state, actions):
        # Reorder pixel layout to channel-first (N, 1, 84, 84) to match SB3 observations
        self.pixels = np.transpose(pixels, (0, 3, 1, 2))
        self.state = state
        self.actions = actions

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx):
        return {
            "pixels": torch.as_tensor(self.pixels[idx], dtype=torch.uint8),
            "state": torch.as_tensor(self.state[idx], dtype=torch.float32),
            "action": torch.as_tensor(self.actions[idx], dtype=torch.long)
        }

def make_env():
    # Run environment headless during model creation/dummy wrapping
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    env = POPEnv()
    env = Monitor(env)
    return env

def main():
    print("=" * 60)
    print("Behavioral Cloning (Imitation Learning) Training")
    print("=" * 60)

    # 1. Load the demonstration dataset
    dataset_path = os.path.abspath(os.path.join(root_dir, "data", "demonstrations.npz"))
    if not os.path.exists(dataset_path):
        print(f"Error: Demonstration dataset not found at '{dataset_path}'!")
        print("Please run 'python record_demonstrations.py' first to collect expert data.")
        sys.exit(1)

    print(f"Loading expert demonstrations from {dataset_path}...")
    data = np.load(dataset_path)
    pixels = data["pixels"]
    state = data["state"]
    actions = data["actions"]

    num_samples = len(actions)
    print(f"Loaded {num_samples} transitions.")
    
    if num_samples < 64:
        print("Warning: Very small dataset. Please collect more transitions for better results.")

    # Calculate class frequencies and class weights to balance training (inverse frequency)
    classes, counts = np.unique(actions, return_counts=True)
    class_counts = np.zeros(10)
    for c, cnt in zip(classes, counts):
        if c < 10:
            class_counts[c] = cnt
        
    print("\nAction distribution in dataset:")
    action_names = [
        "NONE", "LEFT", "RIGHT", "UP", "DOWN", "SHIFT",
        "LEFT_SHIFT", "RIGHT_SHIFT", "LEFT_UP", "RIGHT_UP"
    ]
    for i, name in enumerate(action_names):
        print(f"  Action {i} ({name:<11}): {int(class_counts[i]):<4} ({class_counts[i]/num_samples*100.0:.2f}%)")
        
    class_weights = num_samples / (10.0 * np.maximum(class_counts, 1.0))
    class_weights = class_weights / np.mean(class_weights)
    print(f"Computed normalized class weights: {np.round(class_weights, 3)}")

    # Create PyTorch DataLoader
    dataset = DemoDataset(pixels, state, actions)
    batch_size = 128
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    # 2. Initialize the environment and the PPO model
    print("\nInitializing PPO environment framework...")
    env = DummyVecEnv([make_env])
    env = VecTransposeImage(env)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Creating PPO model on device: {device}")
    
    # Initialize PPO model with identical hyperparameters to multippo.py
    model = PPO(
        policy="MultiInputPolicy",
        env=env,
        learning_rate=2.5e-4,
        verbose=1,
        device=device
    )

    class_weights_tensor = torch.as_tensor(class_weights, dtype=torch.float32, device=device)

    # 3. Supervised BC Training Loop
    policy = model.policy
    policy.train()

    # Use AdamW to optimize all policy parameters
    optimizer = torch.optim.AdamW(policy.parameters(), lr=5e-4, weight_decay=1e-4)

    epochs = 200
    print(f"\nStarting Balanced Behavioral Cloning training for {epochs} epochs...")
    print(f"Batch size: {batch_size} | Device: {device}\n")

    for epoch in range(1, epochs + 1):
        epoch_losses = []
        epoch_correct = 0
        epoch_total = 0

        for batch in dataloader:
            optimizer.zero_grad()

            # Format batch elements for SB3 obs_to_tensor
            obs_dict = {
                "pixels": batch["pixels"].numpy(),
                "state": batch["state"].numpy()
            }
            
            # Use SB3's robust observation tensor formatter (handles type casting, normalizations, device movement)
            obs_tensor, _ = policy.obs_to_tensor(obs_dict)
            
            # Predict action distribution
            distribution = policy.get_distribution(obs_tensor)
            
            # Target actions
            target_actions = batch["action"].to(device).long()
            
            # Calculate weighted negative log probability to balance classes
            log_prob = distribution.log_prob(target_actions)
            weights = class_weights_tensor[target_actions]
            loss = -(log_prob * weights).mean()

            # Backpropagation
            loss.backward()
            optimizer.step()

            # Metrics
            epoch_losses.append(loss.item())
            
            # Calculate classification accuracy (argmax of logits)
            preds = distribution.distribution.logits.argmax(dim=-1)
            epoch_correct += (preds == target_actions).sum().item()
            epoch_total += len(target_actions)

        mean_loss = np.mean(epoch_losses)
        accuracy = (epoch_correct / epoch_total) * 100.0
        
        print(f"Epoch {epoch:<2}/{epochs} | Loss: {mean_loss:.4f} | Imitation Accuracy: {accuracy:.2f}%")

    # 4. Save BC Pretrained Model
    models_dir = os.path.abspath(os.path.join(root_dir, "models"))
    os.makedirs(models_dir, exist_ok=True)
    save_path = os.path.join(models_dir, "ppo_bc_pretrained")
    print(f"\nSaving imitation learning model to '{save_path}.zip'...")
    model.save(save_path)
    print(f"✓ Model saved successfully to {save_path}.zip!")

    # Close dummy environment
    env.close()

if __name__ == "__main__":
    main()
