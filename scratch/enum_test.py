from enum import IntEnum

class Option(IntEnum):
    debug = 0

print(Option["debug"])