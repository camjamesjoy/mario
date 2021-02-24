from enum import Enum
from pynput.keyboard import Key

class Keys(Enum):
    UP = 'w'
    LEFT = 'a'
    DOWN = 's'
    RIGHT = 'd'
    SPRINT = Key.shift_l
    JUMP = Key.space
    RESET = 'l'
