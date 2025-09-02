import itertools
import sys
import time
# import threading

def spinner_animation(stop_event):
    """Displays a simple spinner animation in the console."""
    spinner = itertools.cycle(['|', '/', '-', '\\'])
    while not stop_event.is_set():
        sys.stdout.write(next(spinner))  # Write the character
        sys.stdout.flush()               # Flush the output
        sys.stdout.write('\b')           # Move the cursor back
        time.sleep(0.1)
