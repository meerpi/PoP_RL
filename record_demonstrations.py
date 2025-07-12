import os
import sys
import time
import numpy as np

# Ensure root directory is in sys.path
root_dir = os.path.abspath(os.path.dirname(__file__))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# 1. Set POPEnv to run Headless (so we don't open the C game's native window)
os.environ["SDL_VIDEODRIVER"] = "dummy"

from POP_env import POPEnv
from stable_baselines3.common.monitor import Monitor

print("Initializing POPEnv in headless mode for manual control...")
env = POPEnv()
env = Monitor(env)

# 2. Restore/delete SDL_VIDEODRIVER so Pygame can open its GUI window
if "SDL_VIDEODRIVER" in os.environ:
    del os.environ["SDL_VIDEODRIVER"]

import pygame

# Initialize Pygame
pygame.init()
pygame.display.set_caption("Principia: Expert Demonstration Recorder")
screen = pygame.display.set_mode((640, 400))
clock = pygame.time.Clock()

# Transition lists
pixels_list = []
state_list = []
action_list = []

def save_data():
    if len(action_list) == 0:
        print("No steps recorded. Nothing to save.")
        return
        
    # Always resolve the target directory relative to the project root directory
    data_dir = os.path.abspath(os.path.join(root_dir, "data"))
    print(f"\nSaving {len(action_list)} steps of demonstrations...")
    os.makedirs(data_dir, exist_ok=True)
    
    target_file = os.path.join(data_dir, "demonstrations.npz")
    # Save as compressed npz file
    np.savez_compressed(
        target_file,
        pixels=np.array(pixels_list, dtype=np.uint8),
        state=np.array(state_list, dtype=np.float32),
        actions=np.array(action_list, dtype=np.int64)
    )
    print(f"✓ Successfully saved demonstrations to {target_file}!")

def main():
    print("\n" + "=" * 60)
    print("Manual Control & Demonstration Recorder")
    print("=" * 60)
    print("Controls:")
    print("  ← / A     : Move LEFT")
    print("  → / D     : Move RIGHT")
    print("  ↑ / W     : Jump / Climb UP")
    print("  ↓ / S     : Crouch / Climb DOWN")
    print("  SHIFT / SPACE / J : SHIFT (Action / Walk / Fight)")
    print("  [R]       : Commit & Reset Level (saves current steps)")
    print("  [BACKSPACE]/[DELETE] : Discard CURRENT active run & Reset")
    print("  [Z]       : UNDO / Discard the LAST completed run & Reset")
    print("  [ESC]     : Save demonstrations and Exit")
    print("=" * 60 + "\n")

    obs, info = env.reset()
    running = True
    
    # Episode temporary buffers
    ep_pixels = []
    ep_state = []
    ep_actions = []
    last_run_len = 0  # To support discarding the last run right after it ends

    while running:
        # Check Pygame window events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    # Commit current run and reset
                    if len(ep_actions) > 0:
                        print(f"\n--- Committing active run ({len(ep_actions)} steps) & Resetting level ---")
                        pixels_list.extend(ep_pixels)
                        state_list.extend(ep_state)
                        action_list.extend(ep_actions)
                        last_run_len = len(ep_actions)
                        ep_pixels.clear()
                        ep_state.clear()
                        ep_actions.clear()
                    obs, info = env.reset()
                elif event.key == pygame.K_BACKSPACE or event.key == pygame.K_DELETE:
                    # Discard ONLY the current active run
                    steps_thrown = len(ep_actions)
                    print(f"\n✗ Current run discarded! Threw away {steps_thrown} steps and resetting level...")
                    ep_pixels.clear()
                    ep_state.clear()
                    ep_actions.clear()
                    obs, info = env.reset()
                elif event.key == pygame.K_z:
                    # Undo the LAST completed run completely (even if the new run has already started)
                    if last_run_len > 0 and len(action_list) >= last_run_len:
                        print(f"\n✗ Last completed run undone! Successfully removed {last_run_len} steps from the dataset.")
                        del pixels_list[-last_run_len:]
                        del state_list[-last_run_len:]
                        del action_list[-last_run_len:]
                        last_run_len = 0
                        # Also clear the active steps of the current run
                        ep_pixels.clear()
                        ep_state.clear()
                        ep_actions.clear()
                        obs, info = env.reset()
                    else:
                        print("\nNo previous committed run found to undo.")

        # Capture key presses for discrete actions (0=NONE, 1=LEFT, 2=RIGHT, 3=UP, 4=DOWN, 5=SHIFT, 6=LEFT_SHIFT, 7=RIGHT_SHIFT, 8=LEFT_UP, 9=RIGHT_UP)
        keys = pygame.key.get_pressed()
        action = 0  # NONE
        
        # Priority mapping (Optimized for discrete RL controls with composite walk/jump support)
        is_shift = keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] or keys[pygame.K_SPACE] or keys[pygame.K_j]
        is_left = keys[pygame.K_LEFT] or keys[pygame.K_a]
        is_right = keys[pygame.K_RIGHT] or keys[pygame.K_d]
        is_up = keys[pygame.K_UP] or keys[pygame.K_w]
        
        if is_left and is_up:
            action = 8  # JUMP LEFT
        elif is_right and is_up:
            action = 9  # JUMP RIGHT
        elif is_left and is_shift:
            action = 6  # WALK LEFT
        elif is_right and is_shift:
            action = 7  # WALK RIGHT
        elif is_up:
            action = 3
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
            action = 4
        elif is_left:
            action = 1
        elif is_right:
            action = 2
        elif is_shift:
            action = 5

        # Step the environment
        next_obs, reward, terminated, truncated, info = env.step(action)
        
        # Save transition to temporary episode buffers
        ep_pixels.append(obs["pixels"])
        ep_state.append(obs["state"])
        ep_actions.append(action)
        
        obs = next_obs

        # Extract raw high-resolution RGB pixels directly from POPEnv frame_buffer
        # Frame buffer is shape (200, 320, 3)
        raw_pixels = np.frombuffer(env.unwrapped.frame_buffer, dtype=np.uint8).reshape((200, 320, 3))
        
        # Render high-res frames to pygame screen
        surf = pygame.surfarray.make_surface(np.transpose(raw_pixels, (1, 0, 2)))
        scaled_surf = pygame.transform.scale(surf, (640, 400))
        screen.blit(scaled_surf, (0, 0))
        pygame.display.flip()

        # Print metrics in window title
        committed_steps = len(action_list)
        active_steps = len(ep_actions)
        pygame.display.set_caption(
            f"Dataset Steps: {committed_steps:<5} | "
            f"Active Steps: {active_steps:<4} | "
            f"HP: {info['hp']} | "
            f"Sword: {info['has_sword']}"
        )

        if terminated or truncated:
            print(f"\n--- Episode ended naturally! Committing {len(ep_actions)} steps and auto-resetting... ---")
            print("  (Press [BACKSPACE] or [DELETE] if you wish to discard this last run)")
            pixels_list.extend(ep_pixels)
            state_list.extend(ep_state)
            action_list.extend(ep_actions)
            last_run_len = len(ep_actions)
            ep_pixels.clear()
            ep_state.clear()
            ep_actions.clear()
            obs, info = env.reset()

        # Throttle loop to match 15 steps per second (~66ms frame step)
        clock.tick(15)

    pygame.quit()
    save_data()
    env.close()

if __name__ == "__main__":
    main()
