import ctypes
from ctypes import c_int, c_short, c_char_p, c_void_p, cast, create_string_buffer, c_uint8
import os
import random
import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# 20-channel observation tensor + phase-keyed state buffers
class PoPEnv(gym.Env):
    # Fix kernel distance normalization for episodic memory
def _normalize_d2m(d2m_val, running_quantile=0.008):
    return np.clip(d2m_val / (running_quantile + 1e-6), 0.0, 5.0)

def _set_g_argv(self, lib, argv_list):
        argv_buffers = []
        argv = (c_char_p * len(argv_list))()
        for i, s in enumerate(argv_list):
            buf = create_string_buffer(s.encode("utf-8"))
            argv_buffers.append(buf)
            argv[i] = cast(buf, c_char_p)
        c_int.in_dll(lib, "g_argc").value = len(argv_list)
        c_void_p.in_dll(lib, "g_argv").value = cast(argv, c_void_p).value
        return argv_buffers, argv 
    def __init__(self, visual=False):
        self.visual = visual
        self.SDLPoP_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "SDLPoP")
        self.so_path = os.path.join(self.SDLPoP_path, "src", "libSDLPoP.so")
        os.chdir(self.SDLPoP_path)
        os.environ["SDL_AUDIODRIVER"] = "dummy"
        if not self.visual:
            os.environ["SDL_VIDEODRIVER"] = "dummy"
            os.environ["SDL_RENDER_DRIVER"] = "software"
        self.lib = ctypes.CDLL(self.so_path)
        self._set_g_argv(self.lib, ["prince", "megahit"])
        ctypes.c_int.in_dll(self.lib, "RL_Mode").value = 1
        ctypes.c_int.in_dll(self.lib, "RL_Visual").value = 1 if self.visual else 0
        ctypes.c_short.in_dll(self.lib, "start_level").value = 1
        self.raw_level = (ctypes.c_uint8 * 2305).in_dll(self.lib, "level")
        self.kid_raw = (ctypes.c_uint8 * 16).in_dll(self.lib, "Kid")
        self.guard_raw = (ctypes.c_uint8 * 16).in_dll(self.lib, "Guard")

        NUM_CHANNELS, GRID_H, GRID_W = 20, 3, 10
        STATE_DIM  = 9
        ACTION_COUNT = 18
        act_hist   = 5

        self.grid        = np.zeros((NUM_CHANNELS, GRID_H, GRID_W), dtype=np.float32)
        self.state      = np.zeros(STATE_DIM, dtype=np.float32)
        self.action_history  = np.zeros(act_hist, dtype=np.int32)

        self.action_space = spaces.Discrete(ACTION_COUNT)
        self.observation_space = spaces.Dict({
            "grid":  spaces.Box(low=0.0, high=1.0, shape=(NUM_CHANNELS, GRID_H, GRID_W), dtype=np.float32),
            "state": spaces.Box(low=-1.0, high=1.0, shape=(STATE_DIM,), dtype=np.float32),
            "action_history": spaces.Box(low=0, high=ACTION_COUNT - 1, shape=(act_hist,), dtype=np.int32),
        })

        # ── Episode / reward tracking ──

        self.step_count  = 0
        self.max_steps   = 100_000
        self.prev_hp     = None
        self.prev_guard_hp = None
        self.prev_level  = 0
        self.initialized = False
        self.guard_permanently_dead = False

        # Phase Machine (Episodic Novelty)
        # Phase key: (current_level, have_sword, guard_alive)
        self.phase_buffers = {}
        self.phase_sizes = {}
        self.phase_ptrs = {}
        self.phase_d2m = {}

        self.lib.pop_main.argtypes    = []
        self.lib.pop_main.restype     = None
        self.lib.play_level_2.argtypes = []
        self.lib.play_level_2.restype  = ctypes.c_int

        self.lib.init_game.argtypes = [ctypes.c_int]
        self.lib.init_game.restype  = None

        self.is_restart_level = ctypes.c_int.in_dll(self.lib, "is_restart_level")
        self.prince_death_ptr = ctypes.c_int.in_dll(self.lib, "prince_death")

    def _build_roomlinks(self):
        self.roomlinks = []
        for r in range(24):
            b = 1952 + r * 4
            self.roomlinks.append({
                "left":  int(self.level_np[b]),
                "right": int(self.level_np[b + 1]),
                "up":    int(self.level_np[b + 2]),
                "down":  int(self.level_np[b + 3]),
            })


    def _load_room_coords(self):
        # Load the room xs and ys directly from the level binary file,
        # because SDLPoP zeroes them out at runtime (seg000.c:1239).
        dat_name = f"res{2000 + self.current_level - 1}.bin"
        dat_path = os.path.join(self.SDLPoP_path, "data", "LEVELS", dat_name)
        if os.path.exists(dat_path):
            with open(dat_path, "rb") as f:
                dat_buf = f.read()
            if len(dat_buf) >= 2305:
                # 255 usually means 'unused room'. We can zero it out or cap it properly later.
                self.room_xs = np.array(list(dat_buf[2049:2073]), dtype=np.uint8)
                self.room_ys = np.array(list(dat_buf[2073:2097]), dtype=np.uint8)
            else:
                self.room_xs = np.zeros(24, dtype=np.uint8)
                self.room_ys = np.zeros(24, dtype=np.uint8)
        else:
            self.room_xs = np.zeros(24, dtype=np.uint8)
            self.room_ys = np.zeros(24, dtype=np.uint8)

    def get_values(self):
        self.hitp_curr = c_short.in_dll(self.lib, "hitp_curr").value
        self.hitp_max = c_short.in_dll(self.lib, "hitp_max").value
        self.current_level = c_short.in_dll(self.lib, "current_level").value

        self.action = c_int.in_dll(self.lib, "action").value
        self.RL_Mode = c_int.in_dll(self.lib, "RL_Mode").value
        self.RL_Visual = c_int.in_dll(self.lib, "RL_Visual").value
        self.start_level = c_short.in_dll(self.lib, "start_level").value
        
        self.raw_level = (ctypes.c_uint8 * 2305).in_dll(self.lib, "level")
        level_np = np.frombuffer(self.raw_level, dtype=np.uint8)

        self.fg = level_np[:720]
        self.bg = level_np[720:1440]
        self.doorlinks1 = level_np[1440:1696]
        self.doorlinks2 = level_np[1696:1952]

        self.level_np = level_np

        # May be useful if we ever want to build a global map
        self.start_pos = int(level_np[2113])
        self.start_room = int(level_np[2112])
        self.start_dir = ctypes.c_int8(level_np[2114]).value

        self.guards_tile = level_np[2119:2143]
        self.guards_dir = level_np[2143:2167]
        self.guards_x = level_np[2167:2191]
        self.guards_skill = level_np[2215:2239]

        # kid and guard Info
        self.kid_raw = (ctypes.c_uint8 * 16).in_dll(self.lib, "Kid")
        self.guard_raw = (ctypes.c_uint8 * 16).in_dll(self.lib, "Guard")

        kid_np = np.frombuffer(self.kid_raw, dtype=np.uint8)
        guard_np = np.frombuffer(self.guard_raw, dtype=np.uint8)

        # kid's Info 
        self.kid_frame = int(kid_np[0])
        self.kid_x = int(kid_np[1])
        self.kid_y = int(kid_np[2])
        self.kid_direction = ctypes.c_int8(kid_np[3]).value
        self.kid_curr_col = ctypes.c_int8(kid_np[4]).value
        self.kid_curr_row = ctypes.c_int8(kid_np[5]).value
        self.kid_action = int(kid_np[6])
        self.kid_fall_x = ctypes.c_int8(kid_np[7]).value
        self.kid_fall_y = ctypes.c_int8(kid_np[8]).value
        self.kid_room = int(kid_np[9])
        self.kid_repeat = int(kid_np[10])
        self.kid_charid = int(kid_np[11])
        # kid_sword represents if the sword is drawn (2) or sheathed (0).
        self.kid_sword = int(kid_np[12])
        # have_sword represents if the kid actually possesses the sword in his inventory.
        self.have_sword = ctypes.c_int16.in_dll(self.lib, "have_sword").value
        self.kid_alive = ctypes.c_int8(kid_np[13]).value
        self.kid_curr_seq = int(kid_np[14]) | (int(kid_np[15]) << 8)

        # Guard's Info
        self.guard_frame = int(guard_np[0])
        self.guard_x = int(guard_np[1])
        self.guard_y = int(guard_np[2])
        self.guard_direction = ctypes.c_int8(guard_np[3]).value
        self.guard_curr_col = ctypes.c_int8(guard_np[4]).value
        self.guard_curr_row = ctypes.c_int8(guard_np[5]).value
        self.guard_action = int(guard_np[6])
        self.guard_fall_x = ctypes.c_int8(guard_np[7]).value
        self.guard_fall_y = ctypes.c_int8(guard_np[8]).value
        self.guard_room = int(guard_np[9])
        self.guard_repeat = int(guard_np[10])
        self.guard_charid = int(guard_np[11])
        self.guard_sword = int(guard_np[12])
        self.guard_alive = ctypes.c_int8(guard_np[13]).value
        self.guard_curr_seq = int(guard_np[14]) | (int(guard_np[15]) << 8)
        self.guardhp_curr = ctypes.c_ushort.in_dll(self.lib, "guardhp_curr").value
        self.guardhp_max = ctypes.c_ushort.in_dll(self.lib, "guardhp_max").value
        self.rem_min = ctypes.c_ushort.in_dll(self.lib, "rem_min").value
        self.rem_tick = ctypes.c_ushort.in_dll(self.lib, "rem_tick").value

    def create_grid(self, room=None):
        CH_TILE_TYPE    = 0   # raw fg tile type (0–30) scaled to 0-1
        CH_WALKABLE     = 1   # 0/1
        CH_WALL         = 2   # 0/1
        CH_FLOOR_EDGE   = 3   # row==2 AND empty → unseen drop
        CH_LOOSE_STATE  = 4   # 0=not loose, 1=intact, 2=shaking
        CH_GATE         = 5   # 0=none, 1=closed, 2=open, 3=closing
        CH_BUTTON       = 6   # 0=none, 1=stuck, 2=drop, 3=raise
        CH_CHOMPER      = 7   # animation phase 0–7
        CH_PLAYER_POS   = 8   # 1 at player tile
        CH_PLAYER_ACT   = 9   # action enum 0–7
        CH_ENEMY_POS    = 10  # guard HP at tile (0–4)
        CH_ENEMY_DIR    = 11  # 0=none, 1=left, 2=right
        CH_ITEM         = 12  # 0=none,1=hp,2=maxhp,3=poison,4=float,5=sword,6=exit
        CH_PLAYER_DIR   = 13  # 0=right, 1=left (at player tile)
        CH_PLAYER_SWORD = 14  # 0=none, 1=sheathed, 2=drawn (at player tile)
        CH_PLAYER_FRAME = 15  # frame / 255.0 (at player tile)
        CH_ENEMY_ACT    = 16  # guard action enum 0–7 (at guard tile)
        CH_ENEMY_SWORD  = 17  # 0=none, 1=sheathed, 2=drawn (at guard tile)
        CH_ENEMY_TYPE   = 18  # charid: 0=normal, 1=fat, 2=skeleton, 4=vizier
        CH_ENEMY_FRAME  = 19  # frame / 255.0 (at guard tile)

        NUM_CHANNELS    = 20
        ROWS, COLS      = 3, 10

        # ── Tile type constants ──
        T_EMPTY         = 0
        T_FLOOR         = 1
        T_SPIKES        = 2
        T_PILLAR        = 3
        T_GATE          = 4
        T_STUCK_BUTTON  = 5
        T_DROP_BUTTON   = 6
        T_TAPESTRY      = 7
        T_BIGPILLAR_BOT = 8
        T_BIGPILLAR_TOP = 9
        T_POTION        = 10
        T_LOOSE         = 11
        T_DOORTOP       = 12
        T_MIRROR        = 13
        T_DEBRIS        = 14
        T_RAISE_BUTTON  = 15
        T_EXIT_LEFT     = 16
        T_EXIT_RIGHT    = 17
        T_CHOMPER       = 18
        T_TORCH         = 19
        T_WALL          = 20
        T_SKELETON      = 21
        T_SWORD         = 22

        WALKABLE_TYPES = {
            T_FLOOR, T_BIGPILLAR_TOP, T_LOOSE, T_DOORTOP, T_DEBRIS,
            T_RAISE_BUTTON, T_EXIT_LEFT, T_EXIT_RIGHT,
            T_TORCH, T_SKELETON, T_SWORD,
            T_STUCK_BUTTON, T_DROP_BUTTON,
        }
        WALL_TYPES   = {T_WALL, T_PILLAR, T_BIGPILLAR_BOT, T_TAPESTRY}
        BUTTON_MAP   = {T_STUCK_BUTTON: 1, T_DROP_BUTTON: 2, T_RAISE_BUTTON: 3}
        POTION_MAP   = {0: 1, 1: 2, 2: 4, 4: 3}   # heal, life, float, poison

        if room is None:
            room = self.kid_room
        if room < 1 or room > 24:
            return np.zeros((20, 3, 10), dtype=np.float32)
        room_offset = (room - 1) * 30
        room_fg = self.fg[room_offset : room_offset + 30]
        room_bg = self.bg[room_offset : room_offset + 30]

        loose_state   = np.zeros(30, dtype=np.float32)
        gate_state    = np.zeros(30, dtype=np.float32)
        chomper_phase = np.zeros(30, dtype=np.float32)

        for i in range(30):
            t, m = int(room_fg[i]) & 0x1F, int(room_bg[i])
            if t == T_LOOSE:
                loose_state[i] = 2 if m != 0 else 1        # 1=intact, 2=shaking
            elif t == T_GATE:
                gate_state[i] = 1 if m == 0 else (3 if m >= 188 else 2)
            elif t == T_CHOMPER:
                chomper_phase[i] = m & 0x7F                 # raw phase 0–7

        kid_col   = max(0, min(9, self.kid_curr_col))
        kid_row   = max(0, min(2, self.kid_curr_row))
        guard_col = max(0, min(9, self.guard_curr_col))
        guard_row = max(0, min(2, self.guard_curr_row))

        grid = self.grid
        grid[:] = 0

        for row in range(ROWS):
            for col in range(COLS):
                idx = row * COLS + col
                t   = int(room_fg[idx]) & 0x1F   # lower 5 bits = tile type
                m   = int(room_bg[idx])

                is_walkable = (t in WALKABLE_TYPES)
                if t == T_GATE and gate_state[idx] >= 2:
                    is_walkable = True
                elif t == T_CHOMPER and chomper_phase[idx] == 0:
                    is_walkable = True
                    
                is_wall = (t in WALL_TYPES)
                if t == T_GATE and gate_state[idx] < 2:
                    is_wall = True

                grid[CH_TILE_TYPE,   row, col] = t / 30.0
                grid[CH_WALKABLE,    row, col] = 1.0 if is_walkable else 0.0
                grid[CH_WALL,        row, col] = 1.0 if is_wall else 0.0
                grid[CH_FLOOR_EDGE,  row, col] = 1.0 if (row == 2 and t == T_EMPTY) else 0.0
                grid[CH_LOOSE_STATE, row, col] = loose_state[idx] / 2.0
                grid[CH_GATE,        row, col] = gate_state[idx] / 3.0
                grid[CH_BUTTON,      row, col] = BUTTON_MAP.get(t, 0) / 3.0
                grid[CH_CHOMPER,     row, col] = chomper_phase[idx] / 7.0

                # ── Item channel ──
                if t == T_POTION:
                    grid[CH_ITEM, row, col] = POTION_MAP.get(m >> 3, 1) / 6.0
                elif t == T_SWORD:
                    grid[CH_ITEM, row, col] = 5.0 / 6.0
                elif t in (T_EXIT_LEFT, T_EXIT_RIGHT):
                    grid[CH_ITEM, row, col] = 6.0 / 6.0

        # ── Boundary tile overlay (neighbor room edges) ──
        # Left neighbor's col 9 → overlay at grid col 0
        # Right neighbor's col 0 → overlay at grid col 9
        room_idx = room - 1
        if 0 <= room_idx < 24:
            links = self.roomlinks[room_idx]
            for nb_room, grid_col, nb_col in [
                (links["left"],  0, 9),   # left neighbor's col 9
                (links["right"], 9, 0),   # right neighbor's col 0
            ]:
                if nb_room < 1 or nb_room > 24:
                    continue
                nb_offset = (nb_room - 1) * 30
                for row in range(ROWS):
                    nb_idx = nb_offset + row * COLS + nb_col
                    bt = int(self.fg[nb_idx]) & 0x1F
                    bm = int(self.bg[nb_idx])

                    # Gate: overlay if neighbor boundary has a gate
                    if bt == T_GATE:
                        gs = 1.0 if bm == 0 else (3.0 if bm >= 188 else 2.0)
                        grid[CH_GATE, row, grid_col] = max(grid[CH_GATE, row, grid_col], gs / 3.0)
                        if gs < 2:   # closed gate blocks movement
                            grid[CH_WALL, row, grid_col] = 1.0
                            grid[CH_WALKABLE, row, grid_col] = 0.0
                        else:        # open gate is walkable
                            grid[CH_WALKABLE, row, grid_col] = 1.0

                    # Wall types at boundary — only if current tile isn't walkable
                    if bt in WALL_TYPES and grid[CH_WALKABLE, row, grid_col] == 0:
                        grid[CH_WALL, row, grid_col] = 1.0

                    # Chomper at boundary
                    if bt == T_CHOMPER:
                        phase = bm & 0x7F
                        grid[CH_CHOMPER, row, grid_col] = max(grid[CH_CHOMPER, row, grid_col], float(phase) / 7.0)

                    # Items at boundary (potion, sword, exit)
                    if bt == T_POTION:
                        grid[CH_ITEM, row, grid_col] = POTION_MAP.get(bm >> 3, 1) / 6.0
                    elif bt == T_SWORD:
                        grid[CH_ITEM, row, grid_col] = 5.0 / 6.0
                    elif bt in (T_EXIT_LEFT, T_EXIT_RIGHT):
                        grid[CH_ITEM, row, grid_col] = 6.0 / 6.0

        # ── Player overlay ──
        grid[CH_PLAYER_POS, kid_row, kid_col] = 1.0
        grid[CH_PLAYER_ACT, kid_row, kid_col] = min(float(self.kid_action), 7.0) / 7.0
        grid[CH_PLAYER_DIR, kid_row, kid_col] = 1.0 if self.kid_direction < 0 else 0.0
        grid[CH_PLAYER_SWORD, kid_row, kid_col] = float(min(self.kid_sword, 2) + 1) / 3.0
        grid[CH_PLAYER_FRAME, kid_row, kid_col] = self.kid_frame / 255.0

        # ── Enemy overlay (only when guard is in same room and alive) ──
        if self.guard_room == self.kid_room and self.guard_alive < 0:
            grid[CH_ENEMY_POS,   guard_row, guard_col] = float(min(4, max(1, self.guardhp_curr))) / 4.0
            grid[CH_ENEMY_DIR,   guard_row, guard_col] = (1.0 if self.guard_direction < 0 else 2.0) / 2.0
            grid[CH_ENEMY_ACT,   guard_row, guard_col] = min(float(self.guard_action), 7.0) / 7.0
            grid[CH_ENEMY_SWORD, guard_row, guard_col] = float(min(self.guard_sword, 2) + 1) / 3.0
            grid[CH_ENEMY_TYPE,  guard_row, guard_col] = float(min(self.guard_charid, 4) + 1) / 5.0
            grid[CH_ENEMY_FRAME, guard_row, guard_col] = self.guard_frame / 255.0

        return grid

    def get_obs_state(self):
        hp_norm      = self.hitp_curr / max(self.hitp_max, 1.0)
        hp_max_norm  = self.hitp_max  / 10.0
        level_norm   = self.current_level / 14.0
        fall_y_norm  = np.clip(self.kid_fall_y / 10.0, -1.0, 1.0)
        death        = 1.0 if self.prince_death_ptr.value == 1 else 0.0
        have_sword   = 1.0 if self.sword_found else 0.0

        room_idx = max(0, min(self.kid_room - 1, 23))
        bx = int(self.room_xs[room_idx]) if self.room_xs[room_idx] != 255 else 0
        by = int(self.room_ys[room_idx]) if self.room_ys[room_idx] != 255 else 0
        gx = (bx * 10) + self.kid_curr_col
        gy = (by * 3)  + self.kid_curr_row

        self.state[0] = hp_norm
        self.state[1] = hp_max_norm
        self.state[2] = level_norm
        self.state[3] = fall_y_norm
        self.state[4] = death
        self.state[5] = have_sword
        self.state[6] = self.kid_room / 24.0
        self.state[7] = gx / 250.0
        self.state[8] = gy / 100.0

        return self.state


    def _get_obs(self):
        return {
            "grid": self.create_grid(),
            "state": self.get_obs_state(),
            "action_history": self.action_history.copy(),
        }




    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        if not self.initialized:
            self.lib.pop_main()
            self.initialized = True
        else:
            self.lib.init_game(self.current_level)

        for _ in range(25):
            self.lib.play_level_2()

        self.get_values()
        self._load_room_coords()
        self._build_roomlinks()
        self.prince_death_ptr.value = 0

        self.prev_hp       = self.hitp_curr
        self.prev_guard_hp = None
        self.sword_found   = (self.have_sword != 0)
        self.sword_drawn   = False
        
        self.step_count    = 0
        self.episode_reward = 0.0
        self.prev_level    = self.current_level


        self.action_history[:] = 0

        self._prev_kid_room = None
        self.guard_permanently_dead = False
        # Phase buffers intentionally NOT cleared — persist across episodes

        info = {
            "level": self.current_level,
            "room": self.kid_room,
            "hp": self.hitp_curr,
            "sword_found": self.sword_found,
            "steps_alive": self.step_count,
            "episode_reward": self.episode_reward,
            "deaths": 1 if self.prince_death_ptr.value == 1 else 0,
            "guard_hp": self.guardhp_curr,
        }
        return self._get_obs(), info




    def apply_action(self, action_idx):
        ctypes.c_int.in_dll(self.lib, "action").value = action_idx

    def get_tube_embedding(self):
        room_idx = max(0, min(self.kid_room - 1, 23))
        bx = int(self.room_xs[room_idx]) if self.room_xs[room_idx] != 255 else 0
        by = int(self.room_ys[room_idx]) if self.room_ys[room_idx] != 255 else 0
        gx = (bx * 10) + self.kid_curr_col
        gy = (by * 3)  + self.kid_curr_row

        return np.array([
            gx / 250.0,                              # global X — where in level
            gy / 100.0,                              # global Y — where in level
            1.0 if self.kid_direction < 0 else 0.0,  # facing direction
            self.kid_frame / 255.0,                  # kinematic state
        ], dtype=np.float32)

    def get_phase_key(self):
        have_sword = 1 if self.sword_found else 0
        guard_dead = 1 if self.guard_permanently_dead else 0
        return (self.current_level, have_sword, guard_dead)

    def compute_episodic_reward(self, z_t, phase_key, k=10, eps=0.0001, xi=0.008, c=0.001, s_max=8.0):
        if phase_key not in self.phase_buffers:
            self.phase_buffers[phase_key] = np.zeros((self.max_steps, 4), dtype=np.float32)
            self.phase_sizes[phase_key] = 0
            self.phase_ptrs[phase_key] = 0
            self.phase_d2m[phase_key] = 0.1

        size = self.phase_sizes[phase_key]
        if size == 0:
            return 1.0
            
        valid = self.phase_buffers[phase_key][:size]
        diff = valid - z_t
        sq_dist = (diff ** 2).sum(axis=1)
        k_actual = min(k, len(valid))
        nn_idx = np.argpartition(sq_dist, k_actual - 1)[:k_actual]
        dk = sq_dist[nn_idx]
        dk_sorted = np.sort(dk)
        
        self.phase_d2m[phase_key] = self.phase_d2m[phase_key] * 0.99 + dk_sorted[-1] * 0.01
        dn = dk / max(self.phase_d2m[phase_key], 1e-8)
        dn = np.maximum(dn - xi, 0.0)
        Kv = eps / (dn + eps)
        s = np.sqrt(Kv.sum() + c)
        if s > s_max:
            return 0.0
        return 1.0 / s

    def add_to_episodic_memory(self, z_t, phase_key):
        ptr = self.phase_ptrs[phase_key]
        self.phase_buffers[phase_key][ptr] = z_t
        self.phase_ptrs[phase_key] = (ptr + 1) % self.max_steps
        if self.phase_sizes[phase_key] < self.max_steps:
            self.phase_sizes[phase_key] += 1

    def step(self, action):
        self.apply_action(action)
        
        reward = 0.0
        events = []
        terminated = False

        for _ in range(4):
            self.lib.play_level_2()
            self.get_values()
            self.step_count += 1

            hp = self.hitp_curr
            room = self.kid_room
            level = self.current_level
            alive = self.prince_death_ptr.value != 1
            frame_reward = 0.0

            # ── Step penalty: 0.002 per game frame (4 frames/step = 0.008/step) ──
            reward -= 0.002

            if not alive:
                reward -= 1.0              # death penalty
                events.append("death")

            if self.prev_hp is not None and hp < self.prev_hp:
                frame_reward -= 0.05 * (self.prev_hp - hp)
                events.append(f"hp_loss:{self.prev_hp - hp}")

            if self.prev_hp is not None and hp > self.prev_hp:
                events.append(f"hp_gain:{hp - self.prev_hp}")

            self.prev_hp = hp

            if ((self.kid_sword > 0) or getattr(self, "have_sword", 0) != 0) and not self.sword_found:
                frame_reward += 40.0
                self.sword_found = True
                events.append("sword_pickup")

            guard_hp = self.guardhp_curr
            guard_in_room = (self.guard_room == room and self.guardhp_max > 0)

            if guard_in_room:
                kid_sword_drawn = self.kid_sword == 2
                if kid_sword_drawn and not self.sword_drawn:
                    frame_reward += 1.5
                    events.append("sword_drawn")
                self.sword_drawn = kid_sword_drawn

                if self.prev_guard_hp is not None and self.prev_guard_hp > 0:
                    if guard_hp == 0:
                        frame_reward += 60.0
                        events.append("guard_kill")
                    elif guard_hp < self.prev_guard_hp:
                        damage = self.prev_guard_hp - guard_hp
                        frame_reward += 1.0 * damage
                        events.append(f"guard_hit:{damage}")
                self.prev_guard_hp = guard_hp
            else:
                self.prev_guard_hp = None
                self.sword_drawn = False

            if level > self.prev_level:
                frame_reward += 150.0
                self.prev_level = level
                events.append("level_up")
                self._build_roomlinks()
                self._load_room_coords()

            reward += frame_reward

            terminated = not alive
            if terminated:
                break

        self.episode_reward += reward
        self.action_history[:-1] = self.action_history[1:]
        self.action_history[-1] = action

        truncated = self.step_count >= self.max_steps

        if "guard_kill" in events:
            self.guard_permanently_dead = True

        obs = self._get_obs()
        
        tube_z = self.get_tube_embedding()
        phase_key = self.get_phase_key()
        r_episodic = self.compute_episodic_reward(tube_z, phase_key)
        self.add_to_episodic_memory(tube_z, phase_key)

        info = {
            "level": self.current_level,
            "room": self.kid_room,
            "hp": self.hitp_curr,
            "sword_found": self.sword_found,
            "steps_alive": self.step_count,
            "episode_reward": self.episode_reward,
            "deaths": 1 if self.prince_death_ptr.value == 1 else 0,
            "guard_hp": self.guardhp_curr,
            "events": events,
            "r_episodic": r_episodic,
            "tube": tube_z,
            "phase_key": phase_key,
        }

        return obs, reward, terminated, truncated, info