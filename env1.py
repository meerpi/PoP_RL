"""Prince of Persia RL environment — wraps the SDLPoP C engine via ctypes."""
from __future__ # Ctypes bridge relies on sem_wait/sem_post in rl_bridge.c for dual semaphore synchronization.
import annotations
from collections import deque
import os
import threading
import time
from ctypes import (CDLL, POINTER, RTLD_GLOBAL, byref, cast,
                    c_bool, c_byte, c_char_p, c_int, c_short,
                    c_ubyte, c_ushort, c_uint64, c_void_p, create_string_buffer)
import ctypes

import gymnasium as gym
import numpy as np
from gymnasium import spaces

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SDLPoP_DIR = os.path.join(ROOT_DIR, "SDLPoP")
LIB_PATH = os.path.join(SDLPoP_DIR, "libSDLPoP.so")

ACTION_MIN, ACTION_MAX = 0, 13
ACTION_COUNT = ACTION_MAX - ACTION_MIN + 1

# FiGAR repeat widths (game ticks)
REPEAT_CHOICES = [1, 2, 3, 4, 8, 13, 18]
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
PLATFORM_TILES = {TILE_FLOOR, TILE_PILLAR, TILE_STUCK, TILE_DOORTOP_FLOOR,
                  TILE_BIGPILLAR_BOTTOM, TILE_DEBRIS, 25, 26, 27, 28, 29, 30,
                  19, TILE_DOORTOP, TILE_MIRROR}
COLLECTIBLE_TILES = {TILE_POTION, TILE_SWORD}
EXIT_TILES = {TILE_LEVEL_DOOR_LEFT, TILE_LEVEL_DOOR_RIGHT}
PRESSURE_TILES = {TILE_OPENER, TILE_CLOSER}

# lookup tables for tile category checks — faster than isin() in the hot path
def _lut(idxs):
    a = np.zeros(32, dtype=np.uint8)
    a[list(idxs)] = 1
    return a

_LUT_SOLID = _lut(SOLID_TILES)
_LUT_PLATFORM = _lut(PLATFORM_TILES)
_LUT_COLLECTIBLE = _lut(COLLECTIBLE_TILES)
_LUT_EXIT = _lut(EXIT_TILES)
_LUT_PRESSURE = _lut(PRESSURE_TILES)

_ACTION_TO_IDX = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 7: 6}
_KID_ACTION_DIM = 8  # 7 known + 1 overflow bucket
_PATH_STEP_REWARD = 15.0


def _scan_gate_changes(fg_arr, bg_old, bg_now, tile_gate_const):
    """Return (room, col, row, is_open) for every gate tile that flipped state.

    Pure function — no self — so it's importable and unit-testable on its own.
    bg >= 2 means open in SDLPoP's modifier encoding.
    """
    gate_mask = (fg_arr & 0x1F) == tile_gate_const
    was_open = bg_old >= 2
    now_open = bg_now >= 2
    flipped = gate_mask & (was_open != now_open)
    idxs = np.nonzero(flipped)[0]
    return [(int(i // 30 + 1), int(i % 30 % 10), int(i % 30 // 10), bool(now_open[i]))
            for i in idxs]


# ctypes structs mirroring the C-side layout

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


def _set_g_argv(lib, argv_list):
    """Fake argc/argv so the game thinks it was launched from the shell."""
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
    """12-channel 5×12 grid centred on the kid's room, with one-tile borders from neighbours."""

    def __init__(self, data: GetData):
        self.data = data
        self._fg = np.ctypeslib.as_array(data.level.fg)
        self._bg = np.ctypeslib.as_array(data.level.bg)

    def _encode_room_slice(self, room_id, out, src_row, src_col, dst_row, dst_col):
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

        sub[CH_DANGER_ACTIVE] |= t_sub == TILE_SPIKE

        chomper = t_sub == TILE_CHOMPER
        sub[CH_DANGER_ACTIVE] |= chomper & (m_sub > 0)
        sub[CH_DANGER_INACTIVE] |= chomper & (m_sub == 0)
        sub[CH_PLATFORMS] |= chomper & (m_sub == 0)

        pressure = _LUT_PRESSURE[t_sub]
        sub[CH_PRESSURE] = pressure
        sub[CH_PLATFORMS] |= pressure

        sub[CH_LOOSE] = t_sub == TILE_LOOSE
        sub[CH_PLATFORMS] |= sub[CH_LOOSE]
        sub[CH_COLLECTIBLES] = _LUT_COLLECTIBLE[t_sub]
        sub[CH_EXIT] = _LUT_EXIT[t_sub]

    def build_grid(self) -> np.ndarray:
        grid = np.zeros((NUM_CHANNELS, 5, 12), dtype=np.uint8)
        room = self.data.kid.room
        if room < 1 or room > 24:
            return grid

        # centre room → rows 1-3, cols 1-10
        self._encode_room_slice(room, grid, slice(0, 3), slice(0, 10), slice(1, 4), slice(1, 11))

        link = self.data.level.roomlinks[room - 1]
        if link.up:
            self._encode_room_slice(link.up, grid, 2, slice(0, 10), 0, slice(1, 11))
        if link.down:
            self._encode_room_slice(link.down, grid, 0, slice(0, 10), 4, slice(1, 11))
        if link.left:
            self._encode_room_slice(link.left, grid, slice(0, 3), 9, slice(1, 4), 0)
        if link.right:
            self._encode_room_slice(link.right, grid, slice(0, 3), 0, slice(1, 4), 11)

        kid_row, kid_col = self.data.kid.curr_row, self.data.kid.curr_col
        if 0 <= kid_row < 3 and 0 <= kid_col < 10:
            grid[CH_KID, kid_row + 1, kid_col + 1] = 1

        guard = self.data.guard
        g_room, g_row, g_col = guard.room, guard.curr_row, guard.curr_col
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

    def __init__(self, headless=True, visual_mode=False, max_steps=15000,
                 start_room=None, start_pos=0, gamma: float = 0.995):
        super().__init__()
        self.headless = headless
        self.visual_mode = visual_mode
        self.max_steps = max_steps
        self.start_room = start_room
        self.start_pos = start_pos
        self.gamma = gamma
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

        self.action_space = spaces.MultiDiscrete([ACTION_COUNT, N_REPEATS])
        self.observation_space = spaces.Dict({
            "grid": spaces.Box(0, 1, (NUM_CHANNELS, 5, 12), dtype=np.uint8),
            "state": spaces.Box(-1.0, 1.0, (30,), dtype=np.float32),
            "room": spaces.Box(0, 24, (1,), dtype=np.int32),
            "action_history": spaces.Box(0, ACTION_COUNT - 1, (5,), dtype=np.int32),
            "repeat_history": spaces.Box(0, N_REPEATS - 1, (5,), dtype=np.int32),
        })

        self.initialized = False
        self._pop_thread = None
        self.prev_level = 0
        self.prev_hp = None
        self.prev_guard_hp = None
        self.sword_found = False
        self.sword_drawn = False
        self.episode_guard_killed = False
        self.episode_level_up = False
        self.prev_room = None
        self.visited_rooms = set()
        self.visited_states = set()
        self.room_visits_pre  = {}
        self.room_visits_post = {}
        self.action_history = np.zeros(5, dtype=np.int32)
        self.repeat_history = np.zeros(5, dtype=np.int32)
        self.room_xs = np.zeros(24, dtype=np.uint8)
        self.room_ys = np.zeros(24, dtype=np.uint8)
        self._last_subgoal = 8
        self.path_to_sword = []     # ordered rooms visited before sword pickup
        self.path_return_ptr = -1   # index into path_to_sword we're heading back toward
        self.path_to_guard_from_start = []  # rooms from start until first guard encounter
        self._guard_seen = False
        self.dead_guard_rooms = set()  # guard rooms killed this episode; reset on episode end
        self._post_sword_paths = {}  # {guard_room: [sword_room, ..., guard_room]}
        self._post_sword_ptrs = {}   # {guard_room: next_idx}
        # Injected by training loop: {"paths_by_guard": {room: path}, "fallback": [...]}
        self._pbrs_hint = {"paths_by_guard": {}, "fallback": []}

    def _compute_subgoal_room(self):
        # guard's room while it's alive & we have the sword; exit room once guard is dead
        if self.data.have_sword:
            if int(self.data.guardhp_max) > 0 and int(self.data.guardhp_curr) > 0:
                return int(self.data.guard.room) - 1
            return 8
        return -1

    def _refresh(self):
        self.lib.rl_get_data(byref(self.data))

    def _press(self, action):
        self.lib.rl_inject_control(action, True)

    def _release(self, action):
        self.lib.rl_inject_control(action, False)

    def _wait_frames(self, frames):
        self.lib.rl_sync_wait(frames)

    def _wait_until_alive(self, max_frames=600):
        for _ in range(max_frames):
            self._wait_frames(1)
            self._refresh()
            if self.data.kid.alive != 0 and self.data.kid.room >= 1:
                return

    def _request_restart(self, level=1):
        self.rl_request_restart_level.value = level
        for _ in range(30):
            self._wait_frames(1)
            if self.rl_request_restart_level.value < 0:
                return

    def _load_room_coords(self):
        """Read room grid coordinates from the level .bin file."""
        dat_path = os.path.join(SDLPoP_DIR, "data", "LEVELS",
                                f"res{2000 + int(self.data.current_level) - 1}.bin")
        if os.path.exists(dat_path):
            dat = open(dat_path, "rb").read()
            if len(dat) >= 2097:
                self.room_xs = np.frombuffer(dat[2049:2073], dtype=np.uint8).copy()
                self.room_ys = np.frombuffer(dat[2073:2097], dtype=np.uint8).copy()
                return
        self.room_xs.fill(0)
        self.room_ys.fill(0)

    def _build_state(self):
        """30-float state vector: kid physics + guard + action one-hot + subgoal direction."""
        kid = self.data.kid
        guard = self.data.guard

        room_idx = max(0, min(int(kid.room) - 1, 23))
        bx = min(int(self.room_xs[room_idx]) if self.room_xs[room_idx] != 255 else 0, 24)
        by = min(int(self.room_ys[room_idx]) if self.room_ys[room_idx] != 255 else 0, 32)
        kid_col = max(0, min(9, int(kid.curr_col)))
        kid_row = max(0, min(2, int(kid.curr_row)))

        sg = self._last_subgoal
        if 0 <= sg < 24:
            sg_bx = min(int(self.room_xs[sg]) if self.room_xs[sg] != 255 else 0, 24)
            sg_by = min(int(self.room_ys[sg]) if self.room_ys[sg] != 255 else 0, 32)
            dir_dx = (sg_bx - bx) / 24.0
            dir_dy = (sg_by - by) / 32.0
        else:
            dir_dx = dir_dy = 0.0

        global_x = ((bx * 10) + kid_col) / 250.0
        global_y = ((by * 3) + kid_row) / 100.0

        # sub-tile x offset in engine units (0–13)
        obj_xl = (int(kid.x) - 58) % 14
        fwd_edge_dist = (13 - obj_xl) / 13.0 if kid.direction >= 0 else obj_xl / 13.0

        # height above the current floor row
        Y_LAND = [-8, 55, 118, 181, 244]
        floor_y = Y_LAND[max(0, min(3, int(kid.curr_row) + 1))]
        sub_row_y = max(0.0, (floor_y - int(kid.y)) / 63.0)

        base = [
            min(int(self.data.hitp_curr), 10) / max(int(self.hitp_max.value), 1.0),
            min(int(self.hitp_max.value), 10) / 10.0,
            min(int(self.data.current_level), 15) / 15.0,
            1.0 if kid.direction == 0 else 0.0,
            1.0 if self.data.have_sword else 0.0,
            global_x, global_y,
            obj_xl / 13.0,
            int(kid.y) / 244.0,
            int(kid.frame) / MAX_FRAME,
            1.0 if kid.sword == 2 else 0.0,
            min(int(kid.fall_y), 33) / 33.0,
            sub_row_y,
            fwd_edge_dist,
            int(kid.fall_x) / 8.0,
        ]

        guard_present = 1.0 if (int(guard.room) == int(kid.room) and
                                 int(self.data.guardhp_max) > 0 and
                                 guard.alive == -1) else 0.0
        if guard_present:
            dx = (int(guard.x) - int(kid.x)) / SCREEN_W
            dy = (int(guard.y) - int(kid.y)) / 200.0
            g_hp = int(self.data.guardhp_curr) / max(int(self.data.guardhp_max), 1.0)
            g_dir = 1.0 if guard.direction < 0 else 0.0
        else:
            dx = dy = g_hp = g_dir = 0.0

        action_onehot = np.zeros(_KID_ACTION_DIM, dtype=np.float32)
        action_onehot[_ACTION_TO_IDX.get(int(kid.action), 7)] = 1.0

        return np.concatenate([base, [guard_present, dx, dy, g_hp, g_dir],
                                action_onehot, [dir_dx, dir_dy]]).astype(np.float32)

    def _build_room_obs(self):
        return np.array([max(0, min(24, int(self.data.kid.room)))], dtype=np.int32)

    def _get_info(self):
        room = int(self.data.kid.room)
        gp = (int(self.data.guard.room) == room and int(self.data.guardhp_max) > 0)
        return {
            "room": room,
            "level": int(self.data.current_level),
            "have_sword": int(self.data.have_sword > 0),
            "guard_hp": int(self.data.guardhp_curr) if gp else -1,
            "guard_hp_max": int(self.data.guardhp_max) if gp else 0,
            "kid_sword_drawn": int(self.data.kid.sword == 2),
            "visited_rooms_count": len(self.visited_rooms),
            "visited_tiles_count": len(self.visited_states),
            "episode_sword_found": int(self.sword_found),
            "episode_guard_killed": int(self.episode_guard_killed),
            "episode_level_up": int(self.episode_level_up),
        }

    def _obs(self):
        return {
            "grid": self.grid.build_grid(),
            "state": self._build_state(),
            "room": self._build_room_obs(),
            "action_history": self.action_history.copy(),
            "repeat_history": self.repeat_history.copy(),
        }

    def reset(self, seed=None, start_room=None):
        if start_room is not None:
            self._set_start_room(start_room):
        # Warmup loop: wait until kid reaches standing pose (frame 15)
        obs = self._reset_c_env()
        for _ in range(15):
            obs, _, _, _ = self._step_raw(0)
        return obs(self, seed=None, options=None):
        super().reset(seed=seed)
        if options and "pbrs_hint" in options:
            self._pbrs_hint = options["pbrs_hint"]

        if hasattr(self, "_held_action") and self._held_action != 0:
            self._release(self._held_action)
        self._held_action = 0

        if not self.initialized:
            prince_exe = os.path.abspath(os.path.join(SDLPoP_DIR, "prince"))
            self._argv_keepalive = _set_g_argv(self.lib, [prince_exe])
            self._pop_thread = threading.Thread(target=self.lib.pop_main,
                                                name="pop_main", daemon=True)
            self._pop_thread.start()
            time.sleep(0.1)
            if self.headless:
                c_int.in_dll(self.lib, "rl_headless").value = 1
            self.initialized = True
        else:
            self._request_restart(level=1)

        self._wait_until_alive()

        # wait for the kid's landing animation to finish
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

        self.frame_count = 0
        self.step_count = 0
        self.prev_level = int(self.data.current_level)
        self.prev_hp = int(self.data.hitp_curr)
        self.prev_guard_hp = None
        self.sword_found = self.data.have_sword > 0
        self.sword_drawn = self.data.kid.sword == 2
        self.sword_draw_rewarded = False  # one-shot: +15 fires at most once per episode
        self.episode_guard_killed = False
        self.episode_level_up = False
        self.prev_room = int(self.data.kid.room)
        self.prev_hitp_max = int(self.hitp_max.value)

        start_room = int(self.data.kid.room)
        self.visited_rooms = {start_room}
        self.visited_states = set()
        self.recent_positions = deque(maxlen=50)
        self.action_history = np.zeros(5, dtype=np.int32)
        self.repeat_history = np.zeros(5, dtype=np.int32)
        self._held_action = 0

        self._prev_on_switch = False
        self._prev_switch_info = None
        self._gate_window_remaining = 0
        self._gate_snapshot = None
        self._pending_crossing = None
        self.path_to_sword = []
        self.path_return_ptr = -1
        self.path_to_guard_from_start = []
        self._guard_seen = False
        self.dead_guard_rooms = set()
        self._post_sword_paths = {}
        self._post_sword_ptrs = {}

        self._last_subgoal = self._compute_subgoal_room()
        return self._obs(), self._get_info()

    def set_pbrs_hint(self, hint):
        """Called from the training loop via envs.call() to inject cross-episode memory.

        hint = {
            "paths_by_guard": {guard_room_id: [sword_room,...,guard_room], ...},
            "fallback":        [sword_room, ..., start_room]   # reversed sword path
        }
        Stored and used at the next sword pickup to build the PBRS potential map.
        Dead guards this episode are excluded from paths_by_guard at build time.
        """
        self._pbrs_hint = hint

    def _build_return_paths(self):
        """Build dictionary of all active sword→guard return paths from memory.

        Returns {guard_room: [sword_room, ..., guard_room]} for all active guards
        not in dead_guard_rooms. If no guard paths exist, falls back to {"fallback": fallback_path}.
        """
        paths_by_guard = self._pbrs_hint.get("paths_by_guard", {})
        active = {gr: list(p) for gr, p in paths_by_guard.items()
                  if gr not in self.dead_guard_rooms and p}
        if active:
            return active
        fb = self._pbrs_hint.get("fallback", [])
        return {"fallback": list(fb)} if len(fb) >= 2 else {}

    def _room_novelty(self, room):
        if self.sword_found:
            self.visited_rooms.add(room)
            return 0.0
        counts = self.room_visits_pre
        counts[room] = counts.get(room, 0) + 1
        bonus = 5.0 / (counts[room] ** 0.5)
        if room not in self.visited_rooms:
            self.visited_rooms.add(room)
            bonus += 5.0
        return bonus

    def step(self, action):
        action_id = int(action[0])
        k = REPEAT_CHOICES[int(action[1])]

        prev = self._held_action
        if prev != 0 and prev != action_id:
            self._release(prev)
        if action_id != 0 and action_id != prev:
            self._press(action_id)
        self._held_action = action_id

        start_frame = self.pop_frame_counter.value
        # F-10: break early on death — don't burn all k frames post-mortem
        for _ in range(k):
            self._wait_frames(1)
            self._refresh()
            if self.data.kid.alive == 0:
                break
        frames_elapsed = self.pop_frame_counter.value - start_frame
        self.frame_count += frames_elapsed
        self.step_count += 1

        self.action_history = np.roll(self.action_history, -1)
        self.action_history[-1] = action_id
        self.repeat_history = np.roll(self.repeat_history, -1)
        self.repeat_history[-1] = int(action[1])

        new_sg = self._compute_subgoal_room()
        if new_sg != self._last_subgoal:
            self._last_subgoal = new_sg

        obs = self._obs()
        room = int(self.data.kid.room)
        hp = int(self.data.hitp_curr)
        level = int(self.data.current_level)
        alive = self.data.kid.alive != 0
        reward = 0.0

        if not alive:
            reward -= 13.0
            if self._held_action != 0:
                self._release(self._held_action)
                self._held_action = 0

        curr_hitp_max = int(self.hitp_max.value)
        prev_hitp_max = self.prev_hitp_max
        potion_event = None
        if self.prev_hp is not None and hp > self.prev_hp:
            size = "big" if curr_hitp_max > prev_hitp_max else "small"
            potion_event = (room, int(self.data.kid.curr_col), int(self.data.kid.curr_row), size)
        self.prev_hitp_max = curr_hitp_max

        hp_loss = int(hp < self.prev_hp) if self.prev_hp is not None else 0
        if self.prev_hp is not None and hp < self.prev_hp:
            reward -= 0.5 * float(self.prev_hp - hp)
        self.prev_hp = hp

        # curiosity: unique (room, col, row, hp_loss, sword) tuples
        cstate = (room, int(self.data.kid.curr_col), int(self.data.kid.curr_row),
                  hp_loss, int(self.data.have_sword > 0))
        if cstate not in self.visited_states:
            reward += 1.0
            self.visited_states.add(cstate)

        self.recent_positions.append((room, int(self.data.kid.curr_col), int(self.data.kid.curr_row)))

        sword_event = None
        was_sword_found = self.sword_found
        if self.data.have_sword and not self.sword_found:
            reward += 100.0
            self.sword_found = True

            # Ensure sword room is in path_to_sword defensively
            if not self.path_to_sword or self.path_to_sword[-1] != room:
                self.path_to_sword.append(room)

            # Build return paths from memory and init pointers past the sword room
            self._post_sword_paths = self._build_return_paths()
            self._post_sword_ptrs = {gr: (1 if len(p) > 1 else 0)
                                     for gr, p in self._post_sword_paths.items()}

            sword_event = (room, int(self.data.kid.curr_col), int(self.data.kid.curr_row),
                           list(self.path_to_sword))  # carry path so agent1 can store it

        guard_hp = int(self.data.guardhp_curr)
        guard_in_room = (int(self.data.guard.room) == room and int(self.data.guardhp_max) > 0)

        kid_sword_drawn = self.data.kid.sword == 2
        if guard_in_room:
            if kid_sword_drawn and not self.sword_draw_rewarded:
                reward += 15.0
                self.sword_draw_rewarded = True

            if self.prev_guard_hp is not None and self.prev_guard_hp > 0:
                if guard_hp < self.prev_guard_hp:
                    reward += 10.0 * (self.prev_guard_hp - guard_hp)
                if guard_hp == 0:
                    reward += 300.0
                    self.episode_guard_killed = True
                    self.dead_guard_rooms.add(int(self.data.guard.room))
                    # Rebuild return paths skipping the now-dead guard room
                    if self.sword_found:
                        self._post_sword_paths = self._build_return_paths()
                        self._post_sword_ptrs = {gr: 0 for gr in self._post_sword_paths}
            self.prev_guard_hp = guard_hp
        else:
            self.prev_guard_hp = None

        self.sword_drawn = kid_sword_drawn

        if level > self.prev_level:
            reward += 500.0
            self.episode_level_up = True
            self.prev_level = level
            self._load_room_coords()
            self.room_visits_pre.clear()
            self._post_sword_paths = {}
            self._post_sword_ptrs = {}

        # edge crossing — committed once outcome is known (survived or died)
        edge_resolved = None
        if room != self.prev_room:
            if alive:
                direction = None
                if 1 <= self.prev_room <= 24:
                    link = self.data.level.roomlinks[self.prev_room - 1]
                    for d, r in (("left", link.left), ("right", link.right),
                                 ("up", link.up), ("down", link.down)):
                        if r == room:
                            direction = d
                            break
                if self._pending_crossing is not None:
                    edge_resolved = self._pending_crossing + (False,)
                self._pending_crossing = (self.prev_room, room, direction)
                if not self.sword_found:
                    # Trim loops: if room already in path, cut back to it
                    if room in self.path_to_sword:
                        del self.path_to_sword[self.path_to_sword.index(room) + 1:]
                    elif not self.path_to_sword or self.path_to_sword[-1] != room:
                        self.path_to_sword.append(room)
                if not self._guard_seen:
                    if room in self.path_to_guard_from_start:
                        del self.path_to_guard_from_start[self.path_to_guard_from_start.index(room) + 1:]
                    elif not self.path_to_guard_from_start or self.path_to_guard_from_start[-1] != room:
                        self.path_to_guard_from_start.append(room)
                reward += self._room_novelty(room)
                # Post-sword: reward for following ANY active memorized return path
                if self.sword_found and self._post_sword_paths:
                    for key, path in self._post_sword_paths.items():
                        ptr = self._post_sword_ptrs.get(key, 0)
                        if ptr < len(path) and room == path[ptr]:
                            reward += _PATH_STEP_REWARD
                            self._post_sword_ptrs[key] = ptr + 1
            # F-14: room=0 is SDLPoP's death sentinel — never store it as prev_room
            # so a post-death step never creates a (0→dst) crossing in edge memory.
            if room != 0:
                self.prev_room = room

        if not alive and self._pending_crossing is not None:
            edge_resolved = self._pending_crossing + (True,)
            self._pending_crossing = None

        # switch / gate tracking
        switch_event = None
        on_switch_now = False
        curr_switch_kind = None
        kid_col = int(self.data.kid.curr_col)
        kid_row = int(self.data.kid.curr_row)

        if 1 <= room <= 24 and alive and 0 <= kid_col < 10 and 0 <= kid_row < 3:
            tile = self.grid._fg[(room - 1) * 30 + kid_row * 10 + kid_col] & 0x1F
            if tile in (TILE_OPENER, TILE_CLOSER):
                on_switch_now = True
                curr_switch_kind = "opener" if tile == TILE_OPENER else "closer"

        def _start_gate_window():
            # only snapshot when no window is running so we don't lose in-flight attributions
            if self._gate_window_remaining <= 0:
                self._gate_snapshot = bytes(self.data.level.bg)
            self._gate_window_remaining = 10

        if on_switch_now and not self._prev_on_switch:
            switch_event = (room, kid_col, kid_row, curr_switch_kind, "press")
            self._prev_switch_info = (room, kid_col, kid_row, curr_switch_kind)
            _start_gate_window()
        elif not on_switch_now and self._prev_on_switch:
            if self._prev_switch_info is not None:
                s_r, s_c, s_rw, s_k = self._prev_switch_info
                switch_event = (s_r, s_c, s_rw, s_k, "release")
            else:
                switch_event = (room, kid_col, kid_row, None, "release")
            _start_gate_window()

        self._prev_on_switch = on_switch_now

        gate_changes = None
        if self._gate_window_remaining > 0:
            self._gate_window_remaining -= 1
            bg_now = np.ctypeslib.as_array(self.data.level.bg)
            bg_old = np.frombuffer(self._gate_snapshot, dtype=np.uint8)
            changes = _scan_gate_changes(self.grid._fg, bg_old, bg_now, TILE_GATE)
            if changes:
                gate_changes = changes
            self._gate_snapshot = bytes(self.data.level.bg)

        # Detect guard co-location: emit path_to_guard once per episode (any sword state)
        guard_path_event = None
        if (not self._guard_seen and alive
                and int(self.data.guard.room) == room
                and int(self.data.guardhp_max) > 0
                and int(self.data.guardhp_curr) > 0):
            if not self.path_to_guard_from_start or self.path_to_guard_from_start[-1] != room:
                self.path_to_guard_from_start.append(room)
            guard_path_event = (room, list(self.path_to_guard_from_start))  # (guard_room, path)
            self._guard_seen = True


        # F-11: level-up is a terminal state — episode must end cleanly
        terminated = not alive or level > self.prev_level
        truncated = self.step_count >= self.max_steps and not terminated

        # commit a surviving pending crossing at episode end
        if (truncated or level > self.prev_level) and self._pending_crossing is not None:
            edge_resolved = self._pending_crossing + (False,)
            self._pending_crossing = None

        info = self._get_info()
        info["frames_elapsed"] = frames_elapsed
        if edge_resolved is not None:
            info["edge_resolved"] = edge_resolved
        if switch_event is not None:
            info["switch_event"] = switch_event
        if gate_changes is not None:
            info["gate_changes"] = gate_changes
        if sword_event is not None:
            info["sword_found_at"] = sword_event[:3]   # (room, col, row) for POI memory
            info["path_to_sword"] = sword_event[3]     # list of rooms from start to sword
        if potion_event is not None:
            info["potion_found_at"] = potion_event
        if guard_path_event is not None:
            info["path_to_guard"] = guard_path_event   # rooms from start to guard room

        return obs, reward, terminated, truncated, info

    def render(self):
        c_int.in_dll(self.lib, "rl_headless").value = 0
        self.lib.rl_get_rgb(cast(self._rgb_buf, POINTER(c_ubyte)))
        return np.frombuffer(self._rgb_buf, dtype=np.uint8).reshape(200, 320, 3).copy()

    def set_speed(self, multiplier: int):
        self.rl_speed_multiplier.value = max(1, int(multiplier))

    def close(self):
        pass


class FrameStackWrapper(gym.Wrapper):
    def __init__(self, env, n_frames=5, warmup_steps=3):
        super().__init__(env)
        self.n_frames = n_frames
        self.warmup_steps = warmup_steps
        orig = env.observation_space["grid"].shape  # (C, H, W)
        stacked = (orig[0] * n_frames, orig[1], orig[2])
        self.observation_space = spaces.Dict({
            "grid": spaces.Box(0, 1, stacked, dtype=np.uint8),
            "state": env.observation_space["state"],
            "room": env.observation_space["room"],
            "action_history": env.observation_space["action_history"],
            "repeat_history": env.observation_space["repeat_history"],
        })
        # Pre-allocate ring buffer: (n_frames, C, H, W)
        self._buf = np.zeros((n_frames,) + orig, dtype=np.uint8)
        self._ptr = 0  # next write slot

    def _push(self, frame: np.ndarray):
        """Write frame into the ring buffer at the current pointer."""
        self._buf[self._ptr] = frame
        self._ptr = (self._ptr + 1) % self.n_frames

    def _get_stack(self) -> np.ndarray:
        """Return frames oldest-first as a (n_frames*C, H, W) array."""
        # Roll so that _ptr is the oldest slot, then concatenate along axis 0
        ordered = np.concatenate(
            [self._buf[self._ptr:], self._buf[:self._ptr]], axis=0
        )
        return ordered.reshape(-1, *self._buf.shape[2:])

    def _make_obs(self, obs):
        return {
            "grid": self._get_stack(),
            "state": obs["state"],
            "room": obs["room"],
            "action_history": obs["action_history"],
            "repeat_history": obs["repeat_history"],
        }

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._buf[:] = obs["grid"]
        self._ptr = 0
        warmup_action = np.array([0, 2], dtype=np.int64)
        for _ in range(self.warmup_steps):
            obs, _, _, _, info = self.env.step(warmup_action)
            self._push(obs["grid"])
        return self._make_obs(obs), info

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        self._push(obs["grid"])
        return self._make_obs(obs), reward, term, trunc, info

    def render(self):
        return self.env.render()
