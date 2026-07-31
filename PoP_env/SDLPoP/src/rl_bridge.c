/*
 * rl_bridge.c — Minimal RL interface for SDLPoP
 *
 * Provides functions for an external Python process (via ctypes/CDLL)
 * to inject controls, read game state, and synchronize with the game loop.
 *
 * Synchronization uses dual SDL semaphores for lock-step execution:
 *   rl_go_sem   — game waits on this before each frame (Python authorizes)
 *   rl_step_sem — Python waits on this after each frame (game signals done)
 */

#include "common.h"
#include <stdbool.h>
#include <stdint.h>
#include <string.h>

/* ── Global RL state variables (read/written from Python via ctypes) ── */

int RL_state = 0;                  /* 0 = normal, 1 = RL mode */
int rl_headless = 0;               /* 1 = skip SDL rendering */
int rl_request_restart_level = -1; /* set >=0 by Python to request restart */
uint64_t pop_frame_counter = 0;    /* incremented each game frame */
int rl_speed_multiplier = 1;       /* divide wait_time by this to speed up rendering */

/* ── Lock-step synchronization via dual SDL semaphores ── */

static SDL_sem *rl_step_sem = NULL; /* Python waits: game signals frame done */
static SDL_sem *rl_go_sem = NULL;   /* Game waits: Python authorizes frames  */

void rl_init_sync(void) {
  if (rl_step_sem)
    SDL_DestroySemaphore(rl_step_sem);
  if (rl_go_sem)
    SDL_DestroySemaphore(rl_go_sem);
  rl_step_sem = SDL_CreateSemaphore(0);
  rl_go_sem = SDL_CreateSemaphore(0);
  load_global_options();
}

/*
 * rl_pre_frame — called BEFORE play_frame() in the game loop.
 * Gates the game thread: blocks until Python authorizes this frame.
 */
void rl_pre_frame(void) {
  if (rl_go_sem)
    SDL_SemWait(rl_go_sem);
}

/*
 * rl_frame_hook — called AFTER play_frame() in the game loop.
 * Handles restart requests, then signals Python that this frame is done.
 */
void rl_frame_hook(void) {
  /* Handle restart requests */
  if (rl_request_restart_level >= 0) {
    int target = rl_request_restart_level;
    rl_request_restart_level = -1;

    /* Use the same mechanism as Ctrl+A restart */
    stop_sounds();
    current_level = target;
    is_restart_level = 1;
  }

  /* Signal Python: this frame is complete */
  if (rl_step_sem)
    SDL_SemPost(rl_step_sem);
}

/* Destroy the SDL window immediately (called from Python on level-up). */
void rl_destroy_window(void) {
  if (window_) {
    SDL_DestroyWindow(window_);
    window_ = NULL;
  }
}

/*
 * rl_sync_wait — called from Python. Authorizes n_frames, then blocks
 * until all are done. Game runs at full speed for the batch.
 *
 * Flow per call:
 *   1. Post rl_go_sem N times  → game can run N frames without blocking
 *   2. Wait rl_step_sem N times → sleep until all N frames complete
 *   3. Return: game is blocked at rl_pre_frame(), awaiting next batch
 */
void rl_sync_wait(int n_frames) {
  if (!rl_go_sem || !rl_step_sem)
    return;

  /* Authorize the game to run n_frames */
  for (int i = 0; i < n_frames; i++) {
    SDL_SemPost(rl_go_sem);
  }
  /* Wait for all frames to complete */
  for (int i = 0; i < n_frames; i++) {
    SDL_SemWait(rl_step_sem);
  }
}

/* ── rl_push_key_event: synthesize an SDL key event ── */

static void rl_push_key_event(SDL_Scancode scancode, bool pressed,
                              SDL_Keymod mod) {
  SDL_Event event;
  SDL_zero(event);
  event.type = pressed ? SDL_KEYDOWN : SDL_KEYUP;
  event.key.type = event.type;
  event.key.state = pressed ? SDL_PRESSED : SDL_RELEASED;
  event.key.repeat = 0;
  event.key.keysym.scancode = scancode;
  event.key.keysym.sym = SDL_GetKeyFromScancode(scancode);
  event.key.keysym.mod = mod;
  (void)SDL_PushEvent(&event);
}

/* ── rl_inject_control: map RL action enum to SDL key events ── */
/*
 * Action mapping (matches env1.py / proto.h RL_controls enum):
 *   0  = NONE (no-op)     7  = SHIFT+LEFT
 *   1  = UP               8  = SHIFT+RIGHT
 *   2  = DOWN             9  = UP+LEFT
 *   3  = LEFT            10  = UP+RIGHT
 *   4  = RIGHT           11  = DOWN+LEFT
 *   5  = SHIFT+UP        12  = DOWN+RIGHT
 *   6  = SHIFT+DOWN      13  = INTERACT (shift key)
 */

void rl_inject_control(int action, bool pressed) {
  if (RL_state != 1)
    return;

  const SDL_Scancode shift_sc = SDL_SCANCODE_LSHIFT;
  const SDL_Keymod shift_mod = KMOD_LSHIFT;

  switch (action) {
  case 0: /* NONE */
    break;
  case 1: /* UP */
    rl_push_key_event(SDL_SCANCODE_UP, pressed, KMOD_NONE);
    break;
  case 2: /* DOWN */
    rl_push_key_event(SDL_SCANCODE_DOWN, pressed, KMOD_NONE);
    break;
  case 3: /* LEFT */
    rl_push_key_event(SDL_SCANCODE_LEFT, pressed, KMOD_NONE);
    break;
  case 4: /* RIGHT */
    rl_push_key_event(SDL_SCANCODE_RIGHT, pressed, KMOD_NONE);
    break;

  case 5: /* SHIFT+UP */
    if (pressed) {
      rl_push_key_event(shift_sc, true, KMOD_NONE);
      rl_push_key_event(SDL_SCANCODE_UP, true, shift_mod);
    } else {
      rl_push_key_event(SDL_SCANCODE_UP, false, shift_mod);
      rl_push_key_event(shift_sc, false, KMOD_NONE);
    }
    break;
  case 6: /* SHIFT+DOWN */
    if (pressed) {
      rl_push_key_event(shift_sc, true, KMOD_NONE);
      rl_push_key_event(SDL_SCANCODE_DOWN, true, shift_mod);
    } else {
      rl_push_key_event(SDL_SCANCODE_DOWN, false, shift_mod);
      rl_push_key_event(shift_sc, false, KMOD_NONE);
    }
    break;
  case 7: /* SHIFT+LEFT */
    if (pressed) {
      rl_push_key_event(shift_sc, true, KMOD_NONE);
      rl_push_key_event(SDL_SCANCODE_LEFT, true, shift_mod);
    } else {
      rl_push_key_event(SDL_SCANCODE_LEFT, false, shift_mod);
      rl_push_key_event(shift_sc, false, KMOD_NONE);
    }
    break;
  case 8: /* SHIFT+RIGHT */
    if (pressed) {
      rl_push_key_event(shift_sc, true, KMOD_NONE);
      rl_push_key_event(SDL_SCANCODE_RIGHT, true, shift_mod);
    } else {
      rl_push_key_event(SDL_SCANCODE_RIGHT, false, shift_mod);
      rl_push_key_event(shift_sc, false, KMOD_NONE);
    }
    break;

  case 9: /* UP+LEFT */
    if (pressed) {
      rl_push_key_event(SDL_SCANCODE_UP, true, KMOD_NONE);
      rl_push_key_event(SDL_SCANCODE_LEFT, true, KMOD_NONE);
    } else {
      rl_push_key_event(SDL_SCANCODE_LEFT, false, KMOD_NONE);
      rl_push_key_event(SDL_SCANCODE_UP, false, KMOD_NONE);
    }
    break;
  case 10: /* UP+RIGHT */
    if (pressed) {
      rl_push_key_event(SDL_SCANCODE_UP, true, KMOD_NONE);
      rl_push_key_event(SDL_SCANCODE_RIGHT, true, KMOD_NONE);
    } else {
      rl_push_key_event(SDL_SCANCODE_RIGHT, false, KMOD_NONE);
      rl_push_key_event(SDL_SCANCODE_UP, false, KMOD_NONE);
    }
    break;
  case 11: /* DOWN+LEFT */
    if (pressed) {
      rl_push_key_event(SDL_SCANCODE_DOWN, true, KMOD_NONE);
      rl_push_key_event(SDL_SCANCODE_LEFT, true, KMOD_NONE);
    } else {
      rl_push_key_event(SDL_SCANCODE_LEFT, false, KMOD_NONE);
      rl_push_key_event(SDL_SCANCODE_DOWN, false, KMOD_NONE);
    }
    break;
  case 12: /* DOWN+RIGHT */
    if (pressed) {
      rl_push_key_event(SDL_SCANCODE_DOWN, true, KMOD_NONE);
      rl_push_key_event(SDL_SCANCODE_RIGHT, true, KMOD_NONE);
    } else {
      rl_push_key_event(SDL_SCANCODE_RIGHT, false, KMOD_NONE);
      rl_push_key_event(SDL_SCANCODE_DOWN, false, KMOD_NONE);
    }
    break;
  case 13: /* INTERACT (shift/action key) */
    rl_push_key_event(shift_sc, pressed, pressed ? shift_mod : KMOD_NONE);
    break;
  default:
    break;
  }
}

/* ── rl_get_data: copy game state into a flat struct for Python ── */

#pragma pack(push, 1)
typedef struct {
  char_type kid;
  char_type guard;
  char_type opp;
  level_type level_data;
  unsigned short current_level_val;
  unsigned short hitp_curr_val;
  unsigned short have_sword_val;
  unsigned short guardhp_curr_val;
  unsigned short guardhp_max_val;
} rl_get_data_t;
#pragma pack(pop)

void rl_get_data(void *out_ptr) {
  rl_get_data_t *out = (rl_get_data_t *)out_ptr;
  out->kid = Kid;
  out->guard = Guard;
  out->opp = Opp;
  memcpy(&out->level_data, &level, sizeof(level_type));
  out->current_level_val = (unsigned short)current_level;
  out->hitp_curr_val = (unsigned short)hitp_curr;
  out->have_sword_val = (unsigned short)have_sword;
  out->guardhp_curr_val = (unsigned short)guardhp_curr;
  out->guardhp_max_val = (unsigned short)guardhp_max;
}

/* ── rl_set_start_room: teleport kid to a specific room/position ── */

void rl_set_start_room(unsigned char room, unsigned char pos, signed char dir) {
  if (room < 1 || room > 24)
    return;

  Kid.room = room;
  Kid.curr_col = (pos % 10);
  Kid.curr_row = (pos / 10);
  if (dir != 0) {
    Kid.direction = (dir < 0) ? dir_FF_left : dir_0_right;
  }

  /* Update drawn room to follow the kid */
  drawn_room = room;
  next_room = room;
  load_room_links();
  different_room = 1;
}

/* ── rl_get_rgb: extract current frame as RGB24 bytes (320x200x3) ── */

void rl_get_rgb(uint8_t *out_rgb) {
  SDL_Surface *surface = get_final_surface();
  if (!surface || !surface->pixels)
    return;
  SDL_LockSurface(surface);
  SDL_ConvertPixels(
      surface->w, surface->h,
      surface->format->format,
      surface->pixels,
      surface->pitch,
      SDL_PIXELFORMAT_RGB24,
      out_rgb, surface->w * 3
  );
  SDL_UnlockSurface(surface);
}
