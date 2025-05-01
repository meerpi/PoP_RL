#include <stdio.h>
#include "src/types.h"
#include "src/proto.h"
extern level_type level;
int main() {
    load_level(1);
    for(int i=0; i<24; i++) {
        if(level.guards_tile[i] < 30) {
            printf("Guard found in room %d\n", i+1);
        }
    }
    return 0;
}
