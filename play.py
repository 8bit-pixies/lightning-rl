#!/usr/bin/env python
import sys, gym, time
import curses
scr = curses.initscr()
curses.noecho()  


#
# Test yourself as a learning agent! Pass environment name as a command-line argument, for example:
#
# python keyboard_agent.py SpaceInvadersNoFrameskip-v4
#

from env_catch import CatchEnv

env = CatchEnv({})

if not hasattr(env.action_space, 'n'):
    raise Exception('Keyboard agent only supports discrete action spaces')
ACTIONS = env.action_space.n
SKIP_CONTROL = 0    # Use previous control decision SKIP_CONTROL times, that's how you
                    # can test what skip is still usable.

human_agent_action = 0
human_wants_restart = False
human_sets_pause = False

def key_press(key, mod):
    global human_agent_action, human_wants_restart, human_sets_pause
    if key==0xff0d: human_wants_restart = True
    if key==32: human_sets_pause = not human_sets_pause
    a = int( key - ord('0') )
    if a <= 0 or a >= ACTIONS: return
    human_agent_action = a

def key_release(key, mod):
    global human_agent_action
    a = int( key - ord('0') )
    if a <= 0 or a >= ACTIONS: return
    if human_agent_action == a:
        human_agent_action = 0

env.render()
env.reset()

def show_screen():
    scr.clear()
    screen = env.render()
    for idx, el in enumerate(screen):
        scr.addstr(idx, 0, el)
    time.sleep(0.1)
# env.unwrapped.viewer.window.on_key_press = key_press
# env.unwrapped.viewer.window.on_key_release = key_release

def rollout():
    global human_agent_action, human_wants_restart, human_sets_pause
    human_wants_restart = False
    obser = env.reset()
    skip = 0
    total_reward = 0
    total_timesteps = 0
    done = False
    while env.lives > -1:
        show_screen()

        key = scr.getch()        # This blocks (waits) until the time has elapsed,
        show_screen()

        if key in [ord(" "), ord("0"), ord("1"), ord("2")]:
            if key in [ord(" "), ord("0")]:
                a = 0
            elif key == ord("1"):
                a = 1
            elif key == ord("2"):
                a = 2
            _, _, done, _ = env.step(a)
            # time.sleep(0.1)

    show_screen()

rollout()
show_screen()


print("\ncomplete")