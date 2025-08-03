"""Prince of Persia RL Environment - Clean & Simple"""
from __future__ import annotations
from collections import deque

import os
import threading
import time
from ctypes import CDLL, POINTER, RTLD_GLOBAL, byref, cast, c_bool, c_byte, c_char_p, c_int, c_short, c_ubyte, c_ushort, c_uint64, c_void_p, create_string_buffer
import ctypes

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from obs_builder import ObsBuilder, MAX_ADJ

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SDLPoP_DIR = os.path.join(ROOT_DIR, "SDLPoP")
LIB_PATH = os.path.join(SDLPoP_DIR, "libSDLPoP.so")

ACTION_MIN, ACTION_MAX = 0, 13
ACTION_COUNT = ACTION_MAX - ACTION_MIN + 1

# FiGAR: animation-aligned repeat choices (game ticks). Index into this by the
# second element of the MultiDiscrete action. Paper (Sharma 2017) used W={1..30};
# Empirically verified PoP tick durations: tap/parry(1), micro-step/block(2), prep/edge-step(4), stride/turn(8), step/jump(12).
REPEAT_CHOICES = [1, 2, 4, 8, 12]
N_REPEATS = len(REPEAT_CHOICES)

TILE_EMPTY, TILE_FLOOR, TILE_SPIKE, TILE_PILLAR = 0, 1, 2, 3
TILE_GATE, TILE_STUCK, TILE_CLOSER, TILE_DOORTOP_FLOOR = 4, 5, 6, 7
TILE_BIGPILLAR_BOTTOM, TILE_BIGPILLAR_TOP, TILE_POTION, TILE_LOOSE = 8, 9, 10, 11
TILE_DOORTOP, TILE_MIRROR, TILE_DEBRIS, TILE_OPENER = 12, 13, 14, 15
TILE_LEVEL_DOOR_LEFT, TILE_LEVEL_DOOR_RIGHT, TILE_CHOMPER = 16, 17, 18
TILE_WALL, TILE_SKELETON, TILE_SWORD = 20, 21, 22

CH_WALLS, CH_PLATFORMS, CH_GATES_CLOSED, CH_GATES_OPEN = 0, 1, 2, 3
CH_DANGER_ACTIVE, CH_DANGER_INACTIVE, CH_PRESSURE, CH_LOOSE = 4, 5, 6, 7
CH_COLLECTIBLES, CH_EXIT, CH_KID, CH_GUARD = 8, 9, 10, 11
NUM_CHANNELS = 12

SOLID_TILES = {TILE_WALL, TILE_SKELETON, 23, 24, TILE_BIGPILLAR_TOP}
PLATFORM_TILES = {TILE_FLOOR, TILE_PILLAR, TILE_STUCK, TILE_DOORTOP_FLOOR, TILE_BIGPILLAR_BOTTOM,
                  TILE_DEBRIS, 25, 26, 27, 28, 29, 30, 19, TILE_DOORTOP, TILE_MIRROR}
COLLECTIBLE_TILES = {TILE_POTION, TILE_SWORD}
EXIT_TILES = {TILE_LEVEL_DOOR_LEFT, TILE_LEVEL_DOOR_RIGHT}
PRESSURE_TILES = {TILE_OPENER, TILE_CLOSER}

# Precomputed lookup tables (uint8, shape=(32,)) for each tile set.
# Tile types are 5-bit values (0-31 after & 0x1F). Indexing _LUT_X[t_sub]
# is a single vectorized gather — much faster than 5x np.isin() per call.
_LUT_SOLID      = np.zeros(32, dtype=np.uint8); _LUT_SOLID[list(SOLID_TILES)]      = 1
_LUT_PLATFORM   = np.zeros(32, dtype=np.uint8); _LUT_PLATFORM[list(PLATFORM_TILES)] = 1
_LUT_COLLECTIBLE= np.zeros(32, dtype=np.uint8); _LUT_COLLECTIBLE[list(COLLECTIBLE_TILES)] = 1
_LUT_EXIT       = np.zeros(32, dtype=np.uint8); _LUT_EXIT[list(EXIT_TILES)]         = 1
_LUT_PRESSURE   = np.zeros(32, dtype=np.uint8); _LUT_PRESSURE[list(PRESSURE_TILES)] = 1

_ACTION_TO_IDX = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 7: 6}
_KID_ACTION_DIM = 8  # 7 known ground truth + 1 overflow


class CharStruct(ctypes.Structure):
    _pack_ = 1
    _layout_ = "ms"
    _fields_ = [
        ("frame", c_ubyte), ("x", c_ubyte), ("y", c_ubyte), ("direction", c_byte),
        ("curr_col", c_byte), ("curr_row", c_byte), ("action", c_ubyte),
        ("fall_x", c_byte), ("fall_y", c_byte), ("room", c_ubyte),
        ("repeat", c_ubyte), ("charid", c_ubyte), ("sword", c_ubyte),
        ("alive", c_byte), ("curr_seq", c_ushort),
    ]


class LinkType(ctypes.Structure):
    _pack_ = 1
    _layout_ = "ms"
    _fields_ = [("left", c_ubyte), ("right", c_ubyte), ("up", c_ubyte), ("down", c_ubyte)]


class LevelType(ctypes.Structure):
    _pack_ = 1
    _layout_ = "ms"
    _fields_ = [
        ("fg", c_ubyte * 720), ("bg", c_ubyte * 720),
        ("doorlinks1", c_ubyte * 256), ("doorlinks2", c_ubyte * 256),
        ("roomlinks", LinkType * 24), ("used_rooms", c_ubyte),
        ("roomxs", c_ubyte * 24), ("roomys", c_ubyte * 24), ("fill_1", c_ubyte * 15),
        ("start_room", c_ubyte), ("start_pos", c_ubyte), ("start_dir", c_byte),
        ("fill_2", c_ubyte * 4), ("guards_tile", c_ubyte * 24), ("guards_dir", c_ubyte * 24),
        ("guards_x", c_ubyte * 24), ("guards_seq_lo", c_ubyte * 24), ("guards_skill", c_ubyte * 24),
        ("guards_seq_hi", c_ubyte * 24), ("guards_color", c_ubyte * 24), ("fill_3", c_ubyte * 18),
    ]


class GetData(ctypes.Structure):
    _pack_ = 1
    _layout_ = "ms"
    _fields_ = [
        ("kid", CharStruct), ("guard", CharStruct), ("opp", CharStruct),
        ("level", LevelType), ("current_level", c_ushort),
        ("hitp_curr", c_ushort), ("have_sword", c_ushort),
        ("guardhp_curr", c_ushort), ("guardhp_max", c_ushort),
    ]

SCREEN_W = 320.0
MAX_FRAME = 255.0
MAX_SEQ = 65535.0


def compute_pbrs(curr_dist: int, prev_dist: int, gamma: float = 0.99) -> float:
    """Potential-Based Reward Shaping towards sword room."""
    return gamma * (-float(curr_dist)) - (-float(prev_dist))

def _set_g_argv(lib, argv_list):
    """Set up fake argc/argv so the game thinks it was launched from CLI."""
    argv_buffers = []
    argv = (c_char_p * len(argv_list))()
    for i, s in enumerate(argv_list):
        buf = create_string_buffer(s.encode("utf-8"))
        argv_buffers.append(buf)
        argv[i] = cast(buf, c_char_p)
    c_int.in_dll(lib, "g_argc").value = len(argv_list)
    c_void_p.in_dll(lib, "g_argv").value = cast(argv, c_void_p).value
    return argv_buffers, argv


class GridObs:
    """Builds a single-room observation grid with surrounding border tiles (12ch, 5h, 12w)."""
    def __init__(self, data: GetData):
        self.data = data
        self._fg = np.ctypeslib.as_array(data.level.fg)
        self._bg = np.ctypeslib.as_array(data.level.bg)

    def _encode_room_slice(self, room_id: int, out: np.ndarray,
                           src_row: int | slice, src_col: int | slice,
                           dst_row: int | slice, dst_col: int | slice):
        """Write a slice of a room's tile data into the given sub-region of the grid."""
        if room_id < 1 or room_id > 24:
            return

        offset = (room_id - 1) * 30
        tiles = (self._fg[offset:offset + 30] & 0x1F).reshape(3, 10)
        modif = self._bg[offset:offset + 30].reshape(3, 10)

        t_sub = tiles[src_row, src_col]
        m_sub = modif[src_row, src_col]

        sub = out[:, dst_row, dst_col]

        sub[CH_WALLS] = _LUT_SOLID[t_sub]
        sub[CH_PLATFORMS] = _LUT_PLATFORM[t_sub]

        gate_mask = t_sub == TILE_GATE
        sub[CH_GATES_OPEN] = gate_mask & (m_sub >= 2)
        sub[CH_GATES_CLOSED] = gate_mask & (m_sub < 2)
        sub[CH_PLATFORMS] |= sub[CH_GATES_OPEN]

        spike_mask = t_sub == TILE_SPIKE
        sub[CH_DANGER_ACTIVE] |= spike_mask

        chomper_mask = t_sub == TILE_CHOMPER
        sub[CH_DANGER_ACTIVE] |= chomper_mask & (m_sub > 0)
        sub[CH_DANGER_INACTIVE] |= chomper_mask & (m_sub == 0)
        sub[CH_PLATFORMS] |= chomper_mask & (m_sub == 0)

        pressure_mask = _LUT_PRESSURE[t_sub]
        sub[CH_PRESSURE] = pressure_mask
        sub[CH_PLATFORMS] |= pressure_mask

        sub[CH_LOOSE] = t_sub == TILE_LOOSE
        sub[CH_PLATFORMS] |= sub[CH_LOOSE]
        sub[CH_COLLECTIBLES] = _LUT_COLLECTIBLE[t_sub]
        sub[CH_EXIT] = _LUT_EXIT[t_sub]

    def build_grid(self) -> np.ndarray:
        """Build the single-room grid with adjacent boundary tiles (5x12)."""
        grid = np.zeros((NUM_CHANNELS, 5, 12), dtype=np.uint8)
        room = self.data.kid.room
        if room < 1 or room > 24:
            return grid

        # 1. Center room (3x10) -> placed at rows 1-3, columns 1-10
        self._encode_room_slice(room, grid, slice(0, 3), slice(0, 10), slice(1, 4), slice(1, 11))

        # 2. Adjacent borders from up, down, left, right rooms
        link = self.data.level.roomlinks[room - 1]
        if link.up:
            self._encode_room_slice(link.up, grid, 2, slice(0, 10), 0, slice(1, 11))
        if link.down:
            self._encode_room_slice(link.down, grid, 0, slice(0, 10), 4, slice(1, 11))
        if link.left:
            self._encode_room_slice(link.left, grid, slice(0, 3), 9, slice(1, 4), 0)
        if link.right:
            self._encode_room_slice(link.right, grid, slice(0, 3), 0, slice(1, 4), 11)

        # 3. Add Kid & Guard (shift by +1 row/col offset)
        kid_row, kid_col = self.data.kid.curr_row, self.data.kid.curr_col
        if 0 <= kid_row < 3 and 0 <= kid_col < 10:
            grid[CH_KID, kid_row + 1, kid_col + 1] = 1

        guard = self.data.guard
        g_room = guard.room
        g_row, g_col = guard.curr_row, guard.curr_col
        if g_room >= 1 and 0 <= g_row < 3 and 0 <= g_col < 10:
            if g_room == room:
                grid[CH_GUARD, g_row + 1, g_col + 1] = 1
            elif link.left and g_room == link.left and g_col == 9:
                grid[CH_GUARD, g_row + 1, 0] = 1
            elif link.right and g_room == link.right and g_col == 0:
                grid[CH_GUARD, g_row + 1, 11] = 1
            elif link.up and g_room == link.up and g_row == 2:
                grid[CH_GUARD, 0, g_col + 1] = 1
            elif link.down and g_room == link.down and g_row == 0:
                grid[CH_GUARD, 4, g_col + 1] = 1

        return grid


class PoPEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 15}

    def __init__(self, headless=True, visual_mode=False, max_steps=15000, start_room=None, start_pos=0):
        super().__init__()
        self.headless = headless
        self.visual_mode = visual_mode
        self.max_steps = max_steps
        self.start_room = start_room
        self.start_pos = start_pos
        self.frame_count = 0
        self.step_count = 0

        if headless:
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
            os.environ.setdefault("SDL_RENDER_DRIVER", "software")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

        os.chdir(SDLPoP_DIR)
        self.lib = CDLL(LIB_PATH, mode=RTLD_GLOBAL)

        self.lib.pop_main.argtypes = []
        self.lib.pop_main.restype = None
        self.lib.rl_inject_control.argtypes = [c_int, c_bool]
        self.lib.rl_inject_control.restype = None
        self.lib.rl_get_data.argtypes = [POINTER(GetData)]
        self.lib.rl_get_data.restype = None
        self.lib.rl_set_start_room.argtypes = [c_ubyte, c_ubyte, c_byte]
        self.lib.rl_set_start_room.restype = None
        self.lib.rl_sync_wait.argtypes = [c_int]
        self.lib.rl_sync_wait.restype = None
        self.lib.rl_init_sync.argtypes = []
        self.lib.rl_init_sync.restype = None
        self.lib.rl_get_rgb.argtypes = [POINTER(c_ubyte)]
        self.lib.rl_get_rgb.restype = None
        self._rgb_buf = (c_ubyte * (320 * 200 * 3))()

        self.lib.rl_init_sync()
        c_int.in_dll(self.lib, "RL_state").value = 1
        self.rl_request_restart_level = c_int.in_dll(self.lib, "rl_request_restart_level")
        self.pop_frame_counter = c_uint64.in_dll(self.lib, "pop_frame_counter")
        self.rl_speed_multiplier = c_int.in_dll(self.lib, "rl_speed_multiplier")

        c_ubyte.in_dll(self.lib, "enable_info_screen").value = 0
        c_short.in_dll(self.lib, "start_level").value = 1
        self.hitp_max = c_short.in_dll(self.lib, "hitp_max")

        self.data = GetData()
        self.grid = GridObs(self.data)
        self.obs_builder = ObsBuilder(self.data)

        self.action_space = spaces.MultiDiscrete([ACTION_COUNT, N_REPEATS])
        self.observation_space = spaces.Dict({
            "grid": spaces.Box(low=0, high=1, shape=(NUM_CHANNELS, 5, 12), dtype=np.uint8),
            "state": spaces.Box(low=-1.0, high=1.0, shape=(28,), dtype=np.float32),
            "room": spaces.Box(low=0, high=24, shape=(1,), dtype=np.int32),
            "action_history": spaces.Box(low=0, high=ACTION_COUNT - 1, shape=(5,), dtype=np.int32),
            "repeat_history": spaces.Box(low=0, high=N_REPEATS - 1, shape=(5,), dtype=np.int32),
            "graph": spaces.Box(low=0, high=23, shape=(6, MAX_ADJ), dtype=np.uint8),
            "subgoal_hops": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
        })

        self.initialized = False
        self._pop_thread = None
        self.prev_level = 0
        self.prev_hp = None
        self.prev_guard_hp = None
        self.prev_subgoal_hops = None
        self.sword_found = False
        self.sword_drawn = False
        self.prev_room = None
        self.visited_rooms = set()
        self.frontier_rooms = set()
        self.visited_states = set()  # non-episodic: persists across episodes
        self.action_history = np.zeros(5, dtype=np.int32)
        self.repeat_history = np.zeros(5, dtype=np.int32)
        self.room_xs = np.zeros(24, dtype=np.uint8)
        self.room_ys = np.zeros(24, dtype=np.uint8)

    def _refresh(self):
        """Pull latest game state from the C engine."""
        self.lib.rl_get_data(byref(self.data))

    def _press(self, action):
        """Hold down an action key."""
        self.lib.rl_inject_control(action, True)

    def _release(self, action):
        """Release an action key."""
        self.lib.rl_inject_control(action, False)

    def _wait_frames(self, frames):
        """Block until N game frames have elapsed (semaphore sync)."""
        self.lib.rl_sync_wait(frames)

    def _wait_until_alive(self, max_frames=600):
        """Advance frames until the kid is alive and in a valid room."""
        for _ in range(max_frames):
            self._wait_frames(1)
            self._refresh()
            if self.data.kid.alive != 0 and self.data.kid.room >= 1:
                return

    def _request_restart(self, level=1):
        """Ask the game loop to restart at the given level."""
        self.rl_request_restart_level.value = level
        for _ in range(30):
            self._wait_frames(1)
            if self.rl_request_restart_level.value < 0:
                return

    def _load_room_coords(self):
        """Parse layout coordinates (room_xs and room_ys) directly from level bin files."""
        dat_name = f"res{2000 + int(self.data.current_level) - 1}.bin"
        dat_path = os.path.join(SDLPoP_DIR, "data", "LEVELS", dat_name)
        if os.path.exists(dat_path):
            with open(dat_path, "rb") as f:
                dat_buf = f.read()
            if len(dat_buf) >= 2097:
                self.room_xs = np.frombuffer(dat_buf[2049:2073], dtype=np.uint8).copy()
                self.room_ys = np.frombuffer(dat_buf[2073:2097], dtype=np.uint8).copy()
            else:
                self.room_xs.fill(0)
                self.room_ys.fill(0)
        else:
            self.room_xs.fill(0)
            self.room_ys.fill(0)

    def _get_graph_and_hops(self, subgoal_room=8):
        """Fetch graph state and normalized subgoal hops in a single map_graph call."""
        g = self.obs_builder.map_graph(subgoal_room=subgoal_room)
        graph_obs = np.stack([g["edge_src"], g["edge_dst"], g["edge_fatal"], g["edge_risky"],
                              g["edge_trav"], g["edge_mask"]], axis=0).astype(np.uint8)
        hops = g["subgoal_hops"]
        hops_norm = np.array([min(float(hops), 24.0) / 24.0], dtype=np.float32)
        return graph_obs, hops_norm

    def _hops_to_room9(self):
        """BFS hop count from kid's current room to room 9 (0-indexed 8), using cached ObsBuilder value."""
        return self.obs_builder.subgoal_hops

    def _build_state(self):
        """Build the 28-float state vector: 15 base + 5 guard + 8 action one-hot."""
        kid = self.data.kid
        guard = self.data.guard

        room_idx = max(0, min(int(kid.room) - 1, 23))
        bx = min(int(self.room_xs[room_idx]) if self.room_xs[room_idx] != 255 else 0, 24)
        by = min(int(self.room_ys[room_idx]) if self.room_ys[room_idx] != 255 else 0, 32)
        kid_col = max(0, min(9, int(kid.curr_col)))
        kid_row = max(0, min(2, int(kid.curr_row)))

        global_x = ((bx * 10) + kid_col) / 250.0
        global_y = ((by * 3) + kid_row) / 100.0

        # Sub-tile x offset (0–13 within the tile)
        obj_xl = (int(kid.x) - 58) % 14
        # Direction-corrected distance to the forward tile edge (mirrors engine's distance_to_edge)
        # facing right (direction==0): distance = 13 - obj_xl; facing left: distance = obj_xl
        if kid.direction >= 0:  # dir_0_right
            fwd_edge_dist = (13 - obj_xl) / 13.0
        else:                   # dir_FF_left
            fwd_edge_dist = obj_xl / 13.0

        # Sub-row Y offset: how far above the current floor (0 = on floor, 1 = full tile above)
        Y_LAND = [-8, 55, 118, 181, 244]
        floor_y = Y_LAND[max(0, min(3, int(kid.curr_row) + 1))]
        sub_row_y = max(0.0, (floor_y - int(kid.y)) / 63.0)

        base = [
            min(int(self.data.hitp_curr), 10) / max(int(self.hitp_max.value), 1.0),
            min(int(self.hitp_max.value), 10) / 10.0,
            min(int(self.data.current_level), 15) / 15.0,
            1.0 if kid.direction == 0 else 0.0,
            1.0 if self.data.have_sword else 0.0,
            global_x,
            global_y,
            obj_xl / 13.0,          # sub-tile x offset within current tile
            int(kid.y) / 244.0,     # continuous y (floor anchors: 55, 118, 181)
            int(kid.frame) / MAX_FRAME,
            1.0 if kid.sword == 2 else 0.0,
            # --- new features ---
            min(int(kid.fall_y), 33) / 33.0,   # fall speed: 0=grounded, 1=terminal/lethal
            sub_row_y,                          # height above current row's floor
            fwd_edge_dist,                      # distance to forward tile edge (engine formula)
            int(kid.fall_x) / 8.0,             # horizontal jump momentum (signed, ÷8)
        ]

        # Guard features (completely zeroed out if no guard is present in the room)
        # alive == -1 indicates alive state for character enums
        guard_present = 1.0 if (int(guard.room) == int(kid.room) and
                                 int(self.data.guardhp_max) > 0 and
                                 guard.alive == -1) else 0.0

        if guard_present > 0.0:
            dx = (int(guard.x) - int(kid.x)) / SCREEN_W
            dy = (int(guard.y) - int(kid.y)) / 200.0
            g_hp = int(self.data.guardhp_curr) / max(int(self.data.guardhp_max), 1.0)
            g_dir = 1.0 if guard.direction < 0 else 0.0
        else:
            dx = 0.0
            dy = 0.0
            g_hp = 0.0
            g_dir = 0.0

        combat = [guard_present, dx, dy, g_hp, g_dir]

        action_onehot = np.zeros(_KID_ACTION_DIM, dtype=np.float32)
        action_onehot[_ACTION_TO_IDX.get(int(kid.action), 7)] = 1.0

        return np.concatenate([base, combat, action_onehot]).astype(np.float32)

    def _build_room_obs(self):
        return np.array([max(0, min(24, int(self.data.kid.room)))], dtype=np.int32)

    def _get_info(self):
        """Return the info dict exposed to the agent/logger."""
        room = int(self.data.kid.room)
        guard_present = (int(self.data.guard.room) == room and int(self.data.guardhp_max) > 0)
        return {
            "room": room,
            "level": int(self.data.current_level),
            "hp": int(self.data.hitp_curr),
            "have_sword": int(self.data.have_sword > 0),
            "guard_hp": int(self.data.guardhp_curr) if guard_present else -1,
            "guard_hp_max": int(self.data.guardhp_max) if guard_present else 0,
            "kid_sword_drawn": int(self.data.kid.sword == 2),
            "grid_x": int(self.data.kid.curr_col),
            "grid_y": int(self.data.kid.curr_row),
            "alive": self.data.kid.alive != 0,
            "visited_rooms_count": len(self.visited_rooms),
            "visited_tiles_count": len(self.visited_states),
        }

    def reset(self, seed=None, options=None):
        """Reset the episode — restart level, wait for kid to be ready."""
        super().reset(seed=seed)

        if hasattr(self, "_held_action") and self._held_action != 0:
            self._release(self._held_action)
        self._held_action = 0

        if not self.initialized:
            prince_exe = os.path.abspath(os.path.join(SDLPoP_DIR, "prince"))
            self._argv_keepalive = _set_g_argv(self.lib, [prince_exe])
            self._pop_thread = threading.Thread(target=self.lib.pop_main, name="pop_main", daemon=True)
            self._pop_thread.start()
            time.sleep(0.1)
            if self.headless:
                c_int.in_dll(self.lib, "rl_headless").value = 1
            self.initialized = True
        else:
            self._request_restart(level=1)

        self._wait_until_alive()

        for _ in range(120):
            self._wait_frames(1)
            self._refresh()
            if self.data.kid.action == 0:
                break

        if self.start_room is not None:
            self.lib.rl_set_start_room(self.start_room, self.start_pos, 0)
            self._wait_frames(3)

        self._refresh()
        self._load_room_coords()
        self.obs_builder.build_map_graph()
        self.frame_count = 0
        self.step_count = 0
        self.prev_level = int(self.data.current_level)
        self.prev_hp = int(self.data.hitp_curr)
        self.prev_guard_hp = None
        self.sword_found = self.data.have_sword > 0
        self.sword_drawn = self.data.kid.sword == 2
        self.prev_room = int(self.data.kid.room)
        self.prev_subgoal_hops = int(self.obs_builder.subgoal_hops)
        
        start_room = int(self.data.kid.room)
        self.visited_rooms = {start_room}
        self.frontier_rooms = {start_room}
        self.visited_states = set()
        self.recent_positions = deque(maxlen=50)
        self.action_history = np.zeros(5, dtype=np.int32)
        self.repeat_history = np.zeros(5, dtype=np.int32)
        self._held_action = 0

        graph_obs, hops_norm = self._get_graph_and_hops(subgoal_room=8)
        return {"grid": self.grid.build_grid(), "state": self._build_state(), "room": self._build_room_obs(), "action_history": self.action_history.copy(), "repeat_history": self.repeat_history.copy(), "graph": graph_obs, "subgoal_hops": hops_norm}, self._get_info()

    def _neighbors(self, room):
        """Return list of connected room IDs for the given room."""
        if room < 1 or room > 24:
            return []
        link = self.data.level.roomlinks[room - 1]
        return [r for r in (link.left, link.right, link.up, link.down) if r != 0]

    def _update_frontier(self, new_room):
        """Track room discovery and return frontier-based exploration reward."""
        if new_room in self.visited_rooms:
            return 0.0

        self.visited_rooms.add(new_room)
        unexplored = [r for r in self._neighbors(new_room) if r not in self.visited_rooms]
        frontier_gain = len(unexplored)

        if frontier_gain > 0:
            self.frontier_rooms.add(new_room)

        self.frontier_rooms -= {
            r for r in self.frontier_rooms
            if all(n in self.visited_rooms for n in self._neighbors(r))
        }

        return 5.0 * frontier_gain

    def step(self, action):
        """Execute action for k game frames (FiGAR), return (obs, reward, term, trunc, info).

        action is a length-2 array [action_id, k_idx] from the MultiDiscrete space.
        k_idx indexes into REPEAT_CHOICES to get the actual tick count.
        """
        action_id = int(action[0])
        k = REPEAT_CHOICES[int(action[1])]

        prev = self._held_action
        if prev != 0 and prev != action_id:
            self._release(prev)
        if action_id != 0 and action_id != prev:
            self._press(action_id)
        self._held_action = action_id

        start_frame = self.pop_frame_counter.value
        self._wait_frames(k)
        frames_elapsed = self.pop_frame_counter.value - start_frame

        self._refresh()
        self.frame_count += frames_elapsed
        self.step_count += 1

        self.action_history = np.roll(self.action_history, -1)
        self.action_history[-1] = action_id
        self.repeat_history = np.roll(self.repeat_history, -1)
        self.repeat_history[-1] = int(action[1])

        graph_obs, hops_norm = self._get_graph_and_hops(subgoal_room=8)
        obs = {"grid": self.grid.build_grid(), "state": self._build_state(), "room": self._build_room_obs(), "action_history": self.action_history.copy(), "repeat_history": self.repeat_history.copy(), "graph": graph_obs, "subgoal_hops": hops_norm}

        reward = 0.0
        room = int(self.data.kid.room)
        hp = int(self.data.hitp_curr)
        level = int(self.data.current_level)
        alive = self.data.kid.alive != 0

        if not alive:
            reward -= 10.0
            if self._held_action != 0:
                self._release(self._held_action)
                self._held_action = 0

        hp_loss_flag = int(hp < self.prev_hp) if self.prev_hp is not None else 0
        if self.prev_hp is not None and hp < self.prev_hp:
            reward -= 0.5 * (self.prev_hp - hp)
        self.prev_hp = hp

        curiosity_state = (room, int(self.data.kid.curr_col), int(self.data.kid.curr_row), hp_loss_flag, int(self.data.have_sword > 0))
        if curiosity_state not in self.visited_states:
            reward += 1.0
            self.visited_states.add(curiosity_state)

        pos = (room, int(self.data.kid.curr_col), int(self.data.kid.curr_row))
        self.recent_positions.append(pos)

        if self.data.have_sword and not self.sword_found:
            reward += 50.0
            self.sword_found = True

        guard_hp = int(self.data.guardhp_curr)
        guard_in_room = (int(self.data.guard.room) == room and int(self.data.guardhp_max) > 0)

        if guard_in_room:
            kid_sword_drawn = self.data.kid.sword == 2
            if kid_sword_drawn and not self.sword_drawn:
                reward += 15.0
            self.sword_drawn = kid_sword_drawn

            if self.prev_guard_hp is not None and self.prev_guard_hp > 0 and guard_hp < self.prev_guard_hp:
                damage = self.prev_guard_hp - guard_hp
                reward += 10.0 * damage
            if self.prev_guard_hp is not None and self.prev_guard_hp > 0 and guard_hp == 0:
                reward += 300.0
            self.prev_guard_hp = guard_hp
        else:
            self.prev_guard_hp = None
            self.sword_drawn = False

        if level > self.prev_level:
            reward += 500.0
            self.prev_level = level
            self._load_room_coords()

        if room != self.prev_room:
            if alive:
                reward += self._update_frontier(room)
            self.prev_room = room

        current_hops = int(self.obs_builder.subgoal_hops)
        if self.data.have_sword > 0 and self.prev_subgoal_hops is not None:
            if self.prev_subgoal_hops < 9999 and current_hops < 9999:
                hop_diff = self.prev_subgoal_hops - current_hops
                if hop_diff != 0:
                    reward += 15.0 * float(hop_diff)
        self.prev_subgoal_hops = current_hops

        terminated = not alive
        truncated = self.step_count >= self.max_steps

        info = self._get_info()
        info["frames_elapsed"] = frames_elapsed
        return obs, reward, terminated, truncated, info

    def render(self):
        c_int.in_dll(self.lib, "rl_headless").value = 0
        self.lib.rl_get_rgb(cast(self._rgb_buf, POINTER(c_ubyte)))
        return np.frombuffer(self._rgb_buf, dtype=np.uint8).reshape(200, 320, 3).copy()

    def set_speed(self, multiplier: int):
        """Set rendering speed. 1 = real-time, 2 = 2x, etc."""
        self.rl_speed_multiplier.value = max(1, int(multiplier))

    def close(self):
        pass


class FrameStackWrapper(gym.Wrapper):
    def __init__(self, env, n_frames=5, warmup_steps=3):
        super().__init__(env)
        self.n_frames = n_frames
        self.warmup_steps = warmup_steps
        orig = env.observation_space["grid"].shape
        stacked = (orig[0] * n_frames, orig[1], orig[2])
        self.observation_space = spaces.Dict({
            "grid": spaces.Box(low=0, high=1, shape=stacked, dtype=np.uint8),
            "state": env.observation_space["state"],
            "room": env.observation_space["room"],
            "action_history": env.observation_space["action_history"],
            "repeat_history": env.observation_space["repeat_history"],
            "graph": env.observation_space["graph"],
            "subgoal_hops": env.observation_space["subgoal_hops"],
        })
        self.frames = []

    def _stack(self, obs):
        return {"grid": np.concatenate(self.frames, axis=0), "state": obs["state"], "room": obs["room"], "action_history": obs["action_history"], "repeat_history": obs["repeat_history"], "graph": obs["graph"], "subgoal_hops": obs["subgoal_hops"]}

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.frames = [obs["grid"].copy() for _ in range(self.n_frames)]
        # Warmup: action 0 (no-op) with k_idx=2 (9 ticks, middle of range)
        warmup_action = np.array([0, 2], dtype=np.int64)
        for _ in range(self.warmup_steps):
            obs, _, _, _, info = self.env.step(warmup_action)
            self.frames.append(obs["grid"].copy())
            self.frames = self.frames[-self.n_frames:]
        return self._stack(obs), info

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        self.frames.append(obs["grid"].copy())
        self.frames = self.frames[-self.n_frames:]
        return self._stack(obs), reward, term, trunc, info

    def render(self):
        return self.env.render()

