# by Timothy Downs, inputbox written for my map editor

# This program needs a little cleaning up
# It ignores the shift key
# And, for reasons of my own, this program converts "-" to "_"

# A program to get user input, allowing backspace etc
# shown in a box in the middle of the screen
# Called by:
# import inputbox
# answer = inputbox.ask(screen, "Your name")
#
# Only near the center of the screen is blitted to

import pygame, pygame.font, pygame.event, pygame.draw, string
from pygame.locals import *

def get_key():
  while 1:
    event = pygame.event.poll()
    if event.type == KEYDOWN:
      return event.key
    else:
      pass

def display_box(screen, message):
  "Print a message in a box in the middle of the screen"
  fontobject = pygame.font.Font(None,18)
  pygame.draw.rect(screen, (0,0,0),
                   ((screen.get_width() / 2) - 100,
                    (screen.get_height() / 2) - 10,
                    200,20), 0)
  pygame.draw.rect(screen, (255,255,255),
                   ((screen.get_width() / 2) - 102,
                    (screen.get_height() / 2) - 12,
                    204,24), 1)
  if len(message) != 0:
    screen.blit(fontobject.render(message, 1, (255,255,255)),
                ((screen.get_width() / 2) - 100, (screen.get_height() / 2) - 10))
  pygame.display.flip()

def ask(screen, question):
  "ask(screen, question) -> answer"
  pygame.font.init()
  current_string = []
  display_box(screen, question + ": " + string.join(current_string,""))
  while 1:
    inkey = get_key()
    if inkey == K_BACKSPACE:
      current_string = current_string[0:-1]
    elif inkey == K_RETURN:
      break
    elif inkey == K_MINUS:
      current_string.append("_")
    elif inkey <= 127:
      current_string.append(chr(inkey))
    display_box(screen, question + ": " + string.join(current_string,""))
  return string.join(current_string,"")

def ask2(screen, question):
  "ask(screen, question) -> answer"
  pygame.font.init()
  current_string = []
  display_box(screen, question + ": " + string.join(current_string,""))
  while 1:
    inkey = get_key()
    if inkey == K_BACKSPACE:
      if len(current_string) > 0:
        current_string.pop()
    elif inkey == K_RETURN:
      break
    elif inkey == K_ESCAPE:
      return None
    elif inkey == K_KP0:
      current_string.append('0')
    elif inkey == K_KP1:
      current_string.append('1')
    elif inkey == K_KP2:
      current_string.append('2')
    elif inkey == K_KP3:
      current_string.append('3')
    elif inkey == K_KP4:
      current_string.append('4')
    elif inkey == K_KP5:
      current_string.append('5')
    elif inkey == K_KP6:
      current_string.append('6')
    elif inkey == K_KP7:
      current_string.append('7')
    elif inkey == K_KP8:
      current_string.append('8')
    elif inkey == K_KP9:
      current_string.append('9')
    elif inkey == K_2:
      current_string.append('@')
    elif inkey <= 255:
      current_string.append(chr(inkey))
    display_box(screen, question + ": " + string.join(current_string,""))
  return string.join(current_string,"")

def main():
  screen = pygame.display.set_mode((320,240))
  print ask(screen, "Name") + " was entered"

if __name__ == '__main__': main()
