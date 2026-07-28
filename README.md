## Pretrained Checkpoint

The trained model checkpoint is available at `checkpoints/ckpt_final.pt` (iteration 3050).

To evaluate:
```bash
python eval.py --checkpoint_path checkpoints/ckpt_final.pt
```

## Quickstart




# PoP_RL: PPO Agent for Prince of Persia (SDLPoP)
https://github.com/user-attachments/assets/596a974f-ff7c-42e0-8708-664a25bc2aba

### Currently able to get the Fully level up to 2!


## Engine & Environment Design
The environment wraps [SDLPoP](https://github.com/NagyD/SDLPoP), an open-source C disassembly of the original 1989 Prince of Persia.

- **Direct Memory Access**: `PoP_env/wrappers/build_obs.py` uses `ctypes` to read C engine pointers (`seg000.c`, `seg005.c`, `seg006.c`, `seg009.c`), pulling hitpoints, coordinates, room IDs, and tile matrices directly out of process memory.
- **Decision-Frame Input Timing**: Instead of fixed frame-skipping, `is_decision_frame()` hooks into SDLPoP's internal `control()` dispatch loop. The agent is only queried on exact frames where the engine checks user input.
- **FiGAR Action Space**: Action space is `MultiDiscrete([14, 7])` combining 14 discrete actions (`NONE`, movement, combat `SHIFT` grabs, diagonals, `INTERACT`) with 7 repeat choices (`[1, 2, 3, 4, 8, 13, 18]` frames) to handle multi-frame animations like ledge grabs and sword swings.


## Observations & Neural Network Architecture
`Agent` in `ppo.py` fuses multiple spatial and tabular inputs:

- **CoordConv + Dilated Conv2D**: Processes the `12x5x12` room tile grid by appending normalized row/column spatial coordinates, followed by standard 3x3 conv and dilated 3x3 conv (dilation=2) to give filters a 5x5 receptive field without merging tile cells.
- **Learned Embeddings & State Encoder**: Embeds current room ID (8D vector) and 5-step action/repeat history, combined with 29 normalized scalar features (velocities, HP, animation state) into a 128D state representation.
- **Fog-of-War Room Matrix**: A `24x13` global memory table tracking per-room lifetime visits, guard locations, and discovered connectivity graph edges.
- **Dual Value Heads**: Separate `critic` (extrinsic) and `critic_int` (intrinsic curiosity/novelty) heads for dual-stream PPO return estimation.


## Reward Structure
The environment splits signals into extrinsic environment progress and intrinsic exploration:

### Extrinsic Rewards
| Condition | Value |
|---|---|
| Level completion | +500.0 |
| Guard kill | +300.0 |
| Sword pickup | +100.0 |
| Following memorized return path (post-sword) | +15.0 to +30.0 (progress-escalated) |
| Drawing sword near guard | +15.0 |
| Guard HP damage | +10.0 per HP |
| Kid HP damage | -0.5 per HP |
| Death | -10.0 (-5.0 post-sword) |

### Intrinsic Rewards
| Signal | Formula / Condition |
|---|---|
| Room Novelty | `10.0 / sqrt(lifetime_visits) + 5.0 (episodic first-visit)` |
| Curiosity | `+1.0` per unique `(room, col, row, hp_loss, sword_status)` tuple |
