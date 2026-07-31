#include <stdio.h>
#include <stddef.h>
typedef unsigned char byte;
typedef signed char sbyte;
typedef unsigned short word;
typedef struct char_type {
	byte charid;
	sbyte room;
	byte fall_x;
	byte fall_y;
	word curr_seq;
	word frame;
	byte action;
	sbyte direction;
	short x;
	short y;
	short velocity;
	short velocity_y;
	sbyte alive;
	byte sword;
} char_type;
typedef struct {
	byte left;
	byte right;
	byte up;
	byte down;
} roomlinks_type;
typedef struct level_type {
	byte fg[720];
	byte bg[720];
	byte doorlinks1[256];
	byte doorlinks2[256];
	roomlinks_type roomlinks[24];
	byte used_rooms;
	byte roomxs[24];
	byte roomys[24];
	byte fill_1[15];
	byte start_room;
	byte start_pos;
	sbyte start_dir;
	byte fill_2[4];
	byte guards_tile[24];
	byte guards_dir[24];
	byte guards_x[24];
	byte guards_seq_lo[24];
	byte guards_skill[24];
	byte guards_seq_hi[24];
	byte guards_color[24];
	byte fill_3[18];
} level_type;
typedef struct {
  char_type kid;
  char_type guard;
  char_type opp;
  level_type level;
  unsigned short current_level_val;
  unsigned short hitp_curr_val;
  unsigned short have_sword_val;
  unsigned short guardhp_curr_val;
  unsigned short guardhp_max_val;
} GetData;
int main() {
    printf("kid offset: %zu\n", offsetof(GetData, kid));
    printf("guard offset: %zu\n", offsetof(GetData, guard));
    printf("opp offset: %zu\n", offsetof(GetData, opp));
    printf("level offset: %zu\n", offsetof(GetData, level));
    printf("fill_3 offset: %zu\n", offsetof(GetData, level.fill_3));
    printf("current_level offset: %zu\n", offsetof(GetData, current_level_val));
    return 0;
}
