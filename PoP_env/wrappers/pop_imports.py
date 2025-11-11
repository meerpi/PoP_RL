import ctypes
import os

# 1. Define the link_type struct (used for room connections)
class LinkType(ctypes.Structure):
    _fields_ = [
        ("left", ctypes.c_ubyte),
        ("right", ctypes.c_ubyte),
        ("up", ctypes.c_ubyte),
        ("down", ctypes.c_ubyte),
    ]

# 2. Define the exact memory layout of level_type
class SpatialInfo(ctypes.Structure):
    _fields_ = [
        ("fg", ctypes.c_ubyte * 720),         # Foreground tiles for all 24 rooms (30 per room)
        ("bg", ctypes.c_ubyte * 720),         # Background tiles/modifiers
        ("doorlinks1", ctypes.c_ubyte * 256),
        ("doorlinks2", ctypes.c_ubyte * 256),
        ("roomlinks", LinkType * 24),         # The 24 rooms' connections
        ("used_rooms", ctypes.c_ubyte),
        ("roomxs", ctypes.c_ubyte * 24),
        ("roomys", ctypes.c_ubyte * 24),
        ("fill_1", ctypes.c_ubyte * 15),
        ("start_room", ctypes.c_ubyte),
        ("start_pos", ctypes.c_ubyte),
        ("start_dir", ctypes.c_byte),
        ("fill_2", ctypes.c_ubyte * 4),
        ("guards_tile", ctypes.c_ubyte * 24),
        ("guards_dir", ctypes.c_ubyte * 24),
        ("guards_x", ctypes.c_ubyte * 24),
        ("guards_seq_lo", ctypes.c_ubyte * 24),
        ("guards_skill", ctypes.c_ubyte * 24),
        ("guards_seq_hi", ctypes.c_ubyte * 24),
        ("guards_color", ctypes.c_ubyte * 24),
        ("fill_3", ctypes.c_ubyte * 18),
    ]

# 3. Define the character layout
class PlayerInfo(ctypes.Structure):
    _fields_ = [
        ("frame", ctypes.c_ubyte),
        ("x", ctypes.c_ubyte),
        ("y", ctypes.c_ubyte),
        ("direction", ctypes.c_byte),
        ("curr_col", ctypes.c_byte),
        ("curr_row", ctypes.c_byte),
        ("action", ctypes.c_ubyte),
        ("fall_x", ctypes.c_byte),
        ("fall_y", ctypes.c_byte),
        ("room", ctypes.c_ubyte),
        ("repeat", ctypes.c_ubyte),
        ("charid", ctypes.c_ubyte),
        ("sword", ctypes.c_ubyte),
        ("alive", ctypes.c_byte),
        ("curr_seq", ctypes.c_ushort),
    ]

# 4. Interface class
class SDLPoP_Interface:
    def __init__(self, so_path, visual=False):
        # SDL_AUDIODRIVER=dummy: use null audio backend so no real audio device
        # is needed. The dummy driver still fires its audio callback, which drains
        # the sound buffer and clears digi_playing — so the level-start wait loop
        # (while check_sound_playing()) exits normally.
        os.environ["SDL_AUDIODRIVER"] = "dummy"
        os.environ.pop("SDL_VIDEODRIVER", None)
        os.environ.pop("SDL_RENDER_DRIVER", None)

        self.lib = ctypes.CDLL(so_path)
        
        # RL Control Globals added by the patch
        self._rl_step_mode = ctypes.c_int.in_dll(self.lib, "rl_step_mode")
        self._rl_visual_mode = ctypes.c_int.in_dll(self.lib, "rl_visual_mode")
        self._rl_action = ctypes.c_int.in_dll(self.lib, "rl_action")
        self._rl_kid_dead = ctypes.c_int.in_dll(self.lib, "rl_kid_dead")
        
        # Configure modes
        self._rl_step_mode.value = 1
        self._rl_visual_mode.value = 1 if visual else 0
        
        # Set up start_level (MUST be >= 0 to avoid info/splash screen hang)
        self._start_level = ctypes.c_short.in_dll(self.lib, "start_level")
        self._start_level.value = 1
        
        # Set up standard variables
        self._control_x = ctypes.c_byte.in_dll(self.lib, "control_x")
        self._control_y = ctypes.c_byte.in_dll(self.lib, "control_y")
        self._control_shift = ctypes.c_byte.in_dll(self.lib, "control_shift")
        
        self._kid   = PlayerInfo.in_dll(self.lib, "Kid")
        self._guard = PlayerInfo.in_dll(self.lib, "Guard")
        self._hitp_curr = ctypes.c_short.in_dll(self.lib, "hitp_curr")
        self._hitp_max = ctypes.c_ushort.in_dll(self.lib, "hitp_max")
        self._guardhp_curr = ctypes.c_ushort.in_dll(self.lib, "guardhp_curr")
        self._guardhp_max = ctypes.c_ushort.in_dll(self.lib, "guardhp_max")
        self._have_sword = ctypes.c_ushort.in_dll(self.lib, "have_sword")
        self._rem_tick = ctypes.c_ushort.in_dll(self.lib, "rem_tick")
        self._current_level = ctypes.c_ushort.in_dll(self.lib, "current_level")
        self._level = SpatialInfo.in_dll(self.lib, "level")
        self._rem_min = ctypes.c_ushort.in_dll(self.lib, "rem_min")
        
        # Declare C functions
        self.lib.pop_main.argtypes = []
        self.lib.pop_main.restype = None

        self.lib.play_level_2.argtypes = []
        self.lib.play_level_2.restype = ctypes.c_int

        self.lib.init_game.argtypes = [ctypes.c_int]
        self.lib.init_game.restype = None

        self.lib.load_global_options.argtypes = []
        self.lib.load_global_options.restype = None

        # g_argc=1 is the default; seg000.c pop_main() sets g_argv to
        # a safe fake_argv if g_argv is NULL, so no setup needed here.
        self._g_argc = ctypes.c_int.in_dll(self.lib, "g_argc")
        self._g_argc.value = 1

        # Apply global settings from SDLPoP.ini
        self.lib.load_global_options()
