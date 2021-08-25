from abc import ABC, abstractmethod

buffs = ["attack up", "morale up", "speed up"]
debuffs = ["aroma", "stench", "attack down", "morale down", "speed down",
           "fear", "chill", "stun", "poison", "jinx", "fragile", "lethal", "sleep"]


class Buff:

    attack_buff = 0.2
    morale_buff = 0.2
    speed_buff = 0.2

    stacks = ["attack", "poison"]
    turns = ["aroma", "fear", "jinx", "sleep", "chill"]
    stack_turns = ["morale", "speed"]
    binary = ["fragile", "lethal", "stun"]

    def __init__(self):
        self.entries = {
            "attack": 0,  # stacks
            "poison": 0,  # stacks

            "aroma": 0,  # turns
            "fear": 0,  # turns
            "jinx": 0,  # turns
            "sleep": 0,  # turns
            "chill": 0,  # turns

            "morale": {"stacks": 0, "turns": 0},
            "speed": {"stacks": 0, "turns": 0},

            "fragile": False,  # is fragile
            "lethal": False,  # is lethal
            "stun": False,  # is stunned
        }

    def add(self, entry, data):
        if entry in self.stacks:
            self.entries[entry] += data
            return
        if entry in self.turns:
            self.entries


    def get_atk(self):
        pass

    def on_defense(self):
        pass

    def on_turn(self):
        pass
