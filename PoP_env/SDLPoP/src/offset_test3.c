#include <stdio.h>
#include <stddef.h>
#include <SDL2/SDL.h>
#include "types.h"

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

int main() {
    printf("kid offset: %zu\n", offsetof(rl_get_data_t, kid));
    printf("guard offset: %zu\n", offsetof(rl_get_data_t, guard));
    printf("opp offset: %zu\n", offsetof(rl_get_data_t, opp));
    printf("level_data offset: %zu\n", offsetof(rl_get_data_t, level_data));
    printf("current_level_val offset: %zu\n", offsetof(rl_get_data_t, current_level_val));
    printf("sizeof(char_type): %zu\n", sizeof(char_type));
    printf("sizeof(level_type): %zu\n", sizeof(level_type));
    return 0;
}
