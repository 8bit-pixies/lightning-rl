import curses
scr = curses.initscr()
curses.halfdelay(5)           # How many tenths of a second are waited, from 1 to 255
curses.noecho()               # Wont print the input
while True:
    char = scr.getch()        # This blocks (waits) until the time has elapsed,
                              # or there is input to be handled
    scr.clear()               # Clears the screen
    if char != curses.ERR:    # This is true if the user pressed something
        scr.addstr(0, 0, chr(char))
    else:
        scr.addstr(0, 0, "Waiting")