import os
import sys
import threading
import time
from ctypes import (
    CDLL, POINTER, RTLD_GLOBAL, Structure, addressof, byref,
    c_bool, c_byte, c_char_p, c_int, c_short, c_ubyte, c_uint64, c_ushort,
    memmove, pointer, sizeof,
)
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SDLPoP_DIR = os.path.join(ROOT_DIR, "SDLPoP")
LIB_PATH = os.path.join(SDLPoP_DIR, "libSDLPoP.so")


# C-side layout structs from SDLPoP
class CharStruct(Structure):
    _pack_ = 1
    _fields_ = [
        ("frame", c_ubyte), ("x", c_ubyte), ("y", c_ubyte), ("direction", c_byte),
        ("curr_col", c_byte), ("curr_row", c_byte), ("action", c_ubyte),
        ("fall_x", c_byte), ("fall_y", c_byte), ("room", c_ubyte),
        ("repeat", c_ubyte), ("charid", c_ubyte), ("sword", c_ubyte),
        ("alive", c_byte), ("curr_seq", c_ushort),
    ]


class LinkType(Structure):
    _pack_ = 1
    _fields_ = [("left", c_ubyte), ("right", c_ubyte), ("up", c_ubyte), ("down", c_ubyte)]


class LevelType(Structure):
    _pack_ = 1
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


class GetData(Structure):
    _pack_ = 1
    _fields_ = [
        ("kid", CharStruct), ("guard", CharStruct), ("opp", CharStruct),
        ("level", LevelType), ("current_level", c_ushort),
        ("hitp_curr", c_ushort), ("have_sword", c_ushort),
        ("guardhp_curr", c_ushort), ("guardhp_max", c_ushort),
    ]


def _set_g_argv(lib, argv_list):
    # must set g_argv before pop_main or it segfaults reading uninitialised pointers
    argv_type = c_char_p * (len(argv_list) + 1)
    argv_buf = argv_type(*[s.encode("utf-8") for s in argv_list], None)
    g_argv_ptr = POINTER(c_char_p).in_dll(lib, "g_argv")
    c_int.in_dll(lib, "g_argc").value = len(argv_list)
    memmove(addressof(g_argv_ptr), addressof(pointer(argv_buf)), sizeof(POINTER(c_char_p)))
    return argv_buf


# Tile definitions
TILE_EMPTY, TILE_FLOOR, TILE_SPIKE, TILE_PILLAR = 0, 1, 2, 3
TILE_GATE, TILE_STUCK, TILE_CLOSER, TILE_DOORTOP_FLOOR = 4, 5, 6, 7
TILE_BIGPILLAR_BOTTOM, TILE_BIGPILLAR_TOP, TILE_POTION, TILE_LOOSE = 8, 9, 10, 11
TILE_DOORTOP, TILE_MIRROR, TILE_DEBRIS, TILE_OPENER = 12, 13, 14, 15
TILE_LEVEL_DOOR_LEFT, TILE_LEVEL_DOOR_RIGHT, TILE_CHOMPER = 16, 17, 18
TILE_WALL, TILE_SKELETON, TILE_SWORD = 20, 21, 22

# Spatial observation channels
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
PRESSURE_TILES = {TILE_OPENER, TILE_CLOSER, TILE_STUCK}


def _lut(idxs):
    arr = np.zeros(32, dtype=np.uint8)
    arr[list(idxs)] = 1
    return arr


_LUT_SOLID = _lut(SOLID_TILES)
_LUT_PLATFORM = _lut(PLATFORM_TILES)
_LUT_COLLECTIBLE = _lut(COLLECTIBLE_TILES)
_LUT_EXIT = _lut(EXIT_TILES)
_LUT_PRESSURE = _lut(PRESSURE_TILES)

_ACTION_TO_IDX = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 7: 6}


class GridBuilder:

    def __init__(self, data):
        self.data = data

    def build(self):
        fg = np.ctypeslib.as_array(self.data.level.fg)
        bg = np.ctypeslib.as_array(self.data.level.bg)
        grid = np.zeros((NUM_CHANNELS, 5, 12), dtype=np.uint8)

        room = int(self.data.kid.room)
        if room < 1 or room > 24:
            return grid

        def encode_slice(room_id, src_row, src_col, dst_row, dst_col):
            if room_id < 1 or room_id > 24:
                return
            offset = (room_id - 1) * 30
            t_sub = (fg[offset:offset + 30] & 0x1F).reshape(3, 10)[src_row, src_col]
            m_sub = bg[offset:offset + 30].reshape(3, 10)[src_row, src_col]
            sub = grid[:, dst_row, dst_col]

            sub[CH_WALLS] =_LUT_SOLID[t_sub]
            sub[CH_PLATFORMS] =_LUT_PLATFORM[t_sub]

            gate_mask =(t_sub == TILE_GATE)
            sub[CH_GATES_OPEN] =gate_mask & (m_sub >= 2)
            sub[CH_GATES_CLOSED] =gate_mask & (m_sub < 2)
            sub[CH_PLATFORMS] |=sub[CH_GATES_OPEN]

            sub[CH_DANGER_ACTIVE] |=(t_sub == TILE_SPIKE)
            chomper =(t_sub == TILE_CHOMPER)
            sub[CH_DANGER_ACTIVE] |=chomper & (m_sub > 0)
            sub[CH_DANGER_INACTIVE] |= chomper & (m_sub == 0)
            sub[CH_PLATFORMS] |= chomper & (m_sub == 0)

            pressure = _LUT_PRESSURE[t_sub]
            sub[CH_PRESSURE] = pressure
            sub[CH_PLATFORMS] |= pressure

            sub[CH_LOOSE] = (t_sub == TILE_LOOSE)
            sub[CH_PLATFORMS] |= sub[CH_LOOSE]
            sub[CH_COLLECTIBLES] = _LUT_COLLECTIBLE[t_sub]
            sub[CH_EXIT] = _LUT_EXIT[t_sub]

        # Current room (center 3x10 of grid)
        encode_slice(room, slice(0, 3), slice(0, 10), slice(1, 4), slice(1, 11))

        # Adjacent rooms (1-tile border)
        link = self.data.level.roomlinks[room - 1]
        if link.up:
            encode_slice(link.up, 2, slice(0, 10), 0, slice(1, 11))
        if link.down:
            encode_slice(link.down, 0, slice(0, 10), 4, slice(1, 11))
        if link.left:
            encode_slice(link.left, slice(0, 3), 9, slice(1, 4), 0)
        if link.right:
            encode_slice(link.right, slice(0, 3), 0, slice(1, 4), 11)

        # Player
        k_row, k_col = int(self.data.kid.curr_row), int(self.data.kid.curr_col)
        if 0 <= k_row < 3 and 0 <= k_col < 10:
            grid[CH_KID, k_row + 1, k_col + 1] = 1

        # Guard
        guard = self.data.guard
        g_room, g_row, g_col = int(guard.room), int(guard.curr_row), int(guard.curr_col)
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


class ObsBuilder:

    def __init__(self, headless=True):
        self.headless = headless
        if headless:
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
            os.environ.setdefault("SDL_RENDER_DRIVER", "software")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

        os.chdir(SDLPoP_DIR)
        self.lib = CDLL(LIB_PATH, mode=RTLD_GLOBAL)

        # C function signatures
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

        # Initialize C-side sync & variables
        self.lib.rl_init_sync()
        c_int.in_dll(self.lib, "RL_state").value = 1
        self.rl_request_restart_level = c_int.in_dll(self.lib, "rl_request_restart_level")
        self.pop_frame_counter = c_uint64.in_dll(self.lib, "pop_frame_counter")
        self.rl_speed_multiplier = c_int.in_dll(self.lib, "rl_speed_multiplier")

        c_ubyte.in_dll(self.lib, "enable_info_screen").value = 0
        c_short.in_dll(self.lib, "start_level").value = 1
        self._hitp_max = c_short.in_dll(self.lib, "hitp_max")

        self.data = GetData()
        self.grid_builder = GridBuilder(self.data)

        self.initialized = False
        self._pop_thread = None
        self._argv_keepalive = None
        self._held_action = 0

    def init_engine(self):
        if not self.initialized:
            prince_exe = os.path.abspath(os.path.join(SDLPoP_DIR, "prince"))
            self._argv_keepalive = _set_g_argv(self.lib, [prince_exe])
            self._pop_thread = threading.Thread(target=self.lib.pop_main, name="pop_main", daemon=True)
            self._pop_thread.start()
            time.sleep(0.1)
            if self.headless:
                c_int.in_dll(self.lib, "rl_headless").value = 1
            self.initialized = True

    def refresh(self):
        self.lib.rl_get_data(byref(self.data))

    def inject_control(self, action, pressed):
        self.lib.rl_inject_control(action, pressed)

    def press_action(self, action):
        if self._held_action != 0 and self._held_action != action:
            self.inject_control(self._held_action, False)
        if action != 0 and action != self._held_action:
            self.inject_control(action, True)
        self._held_action = action

    def release_held_action(self):
        if self._held_action != 0:
            self.inject_control(self._held_action, False)
            self._held_action = 0

    def sync_wait(self, frames=1):
        self.lib.rl_sync_wait(frames)

    def wait_until_alive(self, max_frames=600):
        for _ in range(max_frames):
            self.sync_wait(1)
            self.refresh()
            if self.data.kid.alive < 0 and self.data.kid.room >= 1:
                return

    def request_restart(self, level=1):
        self.rl_request_restart_level.value = level
        for _ in range(60):
            self.sync_wait(1)
            self.refresh()
            if self.rl_request_restart_level.value < 0:
                return

    def set_start_room(self, room, pos, dir=0):
        self.lib.rl_set_start_room(room, pos, dir)

    def destroy_window(self):
        if hasattr(self.lib, "rl_destroy_window"):
            self.lib.rl_destroy_window()

    def get_rgb(self):
        self.lib.rl_get_rgb(self._rgb_buf)
        return np.ctypeslib.as_array(self._rgb_buf).reshape((200, 320, 3))

    @property
    def kid(self):
        return self.data.kid

    @property
    def hitp_curr(self):
        return int(self.data.hitp_curr)

    @property
    def hitp_max(self):
        return int(self._hitp_max.value)

    @property
    def have_sword(self):
        return bool(self.data.have_sword > 0)

    @property
    def is_dead(self):
        return bool(self.data.kid.alive >= 0)

    def extract_state(self, hitp_max) -> np.ndarray:
        kid = self.data.kid
        guard = self.data.guard

        kid_col = max(0, min(9, int(kid.curr_col)))
        kid_row = max(0, min(2, int(kid.curr_row)))

        obj_xl = (int(kid.x) - 58) % 14
        fwd_edge_dist = (13 - obj_xl) / 13.0 if kid.direction >= 0 else obj_xl / 13.0

        Y_LAND = [-8, 55, 118, 181, 244]
        floor_y = Y_LAND[max(0, min(3, int(kid.curr_row) + 1))]
        sub_row_y = max(0.0, (floor_y - int(kid.y)) / 63.0)

        base = [
            min(int(self.data.hitp_curr), 10) / max(int(hitp_max), 1.0),
            min(int(hitp_max), 10) / 10.0,
            min(int(self.data.current_level), 15) / 15.0,
            1.0 if kid.direction == 0 else 0.0,
            1.0 if self.data.have_sword else 0.0,
            kid_col / 9.0,
            kid_row / 2.0,
            obj_xl / 13.0,
            int(kid.x) / 320.0,
            int(kid.y) / 244.0,
            int(kid.frame) / 255.0,
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
            dx = (int(guard.x) - int(kid.x)) / 320.0
            dy = (int(guard.y) - int(kid.y)) / 200.0
            g_hp = int(self.data.guardhp_curr) / max(int(self.data.guardhp_max), 1.0)
            g_dir = 1.0 if guard.direction < 0 else 0.0
        else:
            dx = dy = g_hp = g_dir = 0.0

        action_onehot = np.zeros(8, dtype=np.float32)
        action_onehot[_ACTION_TO_IDX.get(int(kid.action), 7)] = 1.0

        return np.concatenate([base, [guard_present, dx, dy, g_hp, g_dir],
                                action_onehot]).astype(np.float32)

    def get_obs(self, hitp_max, action_history, repeat_history):
        return {
            "grid": self.grid_builder.build(),
            "state": self.extract_state(hitp_max),
            "room": np.array([max(0, min(24, int(self.data.kid.room)))], dtype=np.int32),
            "action_history": np.array(action_history, dtype=np.int32),
            "repeat_history": np.array(repeat_history, dtype=np.int32),
        }

    def get_info(self):
        kid = self.data.kid
        room = int(kid.room)
        guard_present = (int(self.data.guard.room) == room and int(self.data.guardhp_max) > 0)
        return {
            "room": room,
            "level": int(self.data.current_level),
            "hp": int(self.data.hitp_curr),
            "have_sword": int(self.data.have_sword > 0),
            "guard_hp": int(self.data.guardhp_curr) if guard_present else -1,
            "guard_hp_max": int(self.data.guardhp_max) if guard_present else 0,
            "kid_sword_drawn": int(kid.sword == 2),
            "grid_x": int(kid.curr_col),
            "grid_y": int(kid.curr_row),
            "alive": kid.alive < 0,
        }
