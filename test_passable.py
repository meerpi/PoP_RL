import time
import threading
from worker_env import PoPEnv

class RecorderEnv(PoPEnv):
    def start_recording(self):
        self.recorded_transitions = set()
        self.get_values()
        self.prev_room = self.kid_room
        print("Recording started. Play normally. Close the window or press Ctrl+C to stop.")

    def record_step(self):
        self.get_values()
        curr = self.kid_room
        if curr != self.prev_room:
            self.recorded_transitions.add((self.prev_room, curr))
            print(f"  {self.prev_room} → {curr}")
            self.prev_room = curr

    def stop_recording(self):
        from collections import defaultdict
        graph = defaultdict(list)
        for src, dst in sorted(self.recorded_transitions):
            graph[src].append(dst)
        print("\nLEVEL1_GRAPH = {")
        for room in sorted(graph):
            print(f"    {room}: {sorted(graph[room])},")
        print("}")
        return dict(graph)

if __name__ == "__main__":
    env = RecorderEnv(visual=True)
    # The user wants to play level 1 from the start
    env.reset()
    
    env.start_recording()
    
    stop_event = threading.Event()
    
    def poll_memory():
        while not stop_event.is_set():
            env.record_step()
            time.sleep(0.05)
            
    t = threading.Thread(target=poll_memory)
    t.daemon = True
    t.start()
    
    try:
        # We must allow the user to play. 
        # If rl_mode is 1, keyboard is ignored.
        # If rl_mode is 0, play_level_2 blocks forever but keyboard works.
        import ctypes
        ctypes.c_int.in_dll(env.lib, "rl_mode").value = 0
        # Blocks and runs the native SDL loop
        env.lib.play_level_2()
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        env.stop_recording()
