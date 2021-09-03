from abc import ABC, abstractmethod

buffs = ["attack up", "morale up", "speed up"]
debuffs = ["aroma", "stench", "attack down", "morale down", "speed down",
           "fear", "chill", "stun", "poison", "jinx", "fragile", "lethal", "sleep"]

# sleep before dmg?
# tick fear on OWN actions / own defenses?


class Buffs:
    attack_buff = 0.2
    morale_buff = 0.2
    speed_buff = 0.2
    poison_dmg = 2

    stacks = ["attack up", "attack down", "poison", "fear"]
    turns = ["aroma", "jinx", "sleep", "chill", "stench"]
    stack_turns = ["morale up", "morale down", "speed up", "speed down"]
    binary = ["fragile", "lethal", "stun"]

    def __init__(self):
        self.entries = {
            "attack up": 0,  # stacks
            "attack down": 0,  # stacks
            "poison": 0,  # stacks

            "aroma": 0,  # turns
            "fear": 0,  # turns
            "jinx": 0,  # turns
            "sleep": 0,  # turns
            "chill": 0,  # turns
            "stench": 0,  # turns

            "morale up": {0: 0},  # {turns: stacks, turns: stacks, etc. }
            "morale down": {0: 0},  # {turns: stacks, turns: stacks, etc. }
            "speed up": {0: 0},
            "speed down": {0: 0},

            "fragile": False,  # is fragile
            "lethal": False,  # is lethal
            "stun": False,  # is stunned
        }

    def add(self, entry, data):
        if entry in self.stacks:
            self.entries[entry] += data
            if "attack" in entry:
                self.entries[entry] = max(min(self.entries[entry], 5), -5)
            return
        if entry in self.turns:
            self.entries[entry] = max(self.entries[entry], data)
            return
        if entry in self.stack_turns:
            turns, stacks = data
            if turns in self.entries[entry].keys():
                self.entries[entry][turns] = max(min(self.entries[entry][turns]+stacks, 5), -5)
            else:
                self.entries[entry].update({turns: max(min(stacks, 5), -5)})
            return

    def transfer(self, other):
        # todo
        pass

    def remove_debuffs(self):
        for entry in debuffs:
            if entry in self.stacks:
                self.entries[entry] = 0

            if entry in self.turns:
                self.entries[entry] = 0

            if entry in self.stack_turns:
                self.entries[entry] = dict()

            if entry in self.binary:
                self.entries[entry] = False

    def is_buffed(self):
        for entry in buffs:
            if entry in self.stacks:
                if self.entries[entry] > 0:
                    return True

            if entry in self.turns:
                if self.entries[entry] > 0:
                    return True

            if entry in self.stack_turns and sum(self.entries[entry].values()) > 0:
                return True

            if entry in self.binary:
                if self.entries[entry]:
                    return True

        return False

    def is_debuffed(self):
        for entry in debuffs:
            if entry in self.stacks:
                if self.entries[entry] > 0:
                    return True

            if entry in self.turns:
                if self.entries[entry] > 0:
                    return True

            if entry in self.stack_turns and sum(self.entries[entry].values()) > 0:
                return True

            if entry in self.binary:
                if self.entries[entry]:
                    return True

        return False

    def count_debuffs(self):
        n = 0
        for entry in debuffs:
            if entry in self.stacks:
                if self.entries[entry] > 0:
                    n += self.entries[entry]

            if entry in self.turns:
                if self.entries[entry] > 0:
                    n += 1

            if entry in self.stack_turns:
                n += sum(self.entries[entry].values())

            if entry in self.binary:
                if self.entries[entry]:
                    n += 1

        return n

    def on_determine_order(self):
        return (sum(self.entries["speed up"].values()) - sum(self.entries["speed down"].values())) * self.speed_buff, \
               (sum(self.entries["morale up"].values()) - sum(self.entries["morale down"].values())) * self.morale_buff

    def on_target_select(self):
        """
        :return: focus, evade
        """
        return self.entries["aroma"] > 0, \
               self.entries["stench"] > 0

    def can_hit(self):
        if self.entries["fear"]:
            return False
        if self.entries["stun"]:
            self.entries["stun"] = False
            return False
        return True

    def on_atk(self):
        """
        :return: atk, morale, lethal, jinx
        """
        atk = (self.entries["attack up"] - self.entries["attack down"]) * self.attack_buff
        self.entries["attack up"] = 0
        self.entries["attack down"] = 0

        lethal = self.entries["lethal"]
        self.entries["lethal"] = False

        morale = sum(self.entries["morale up"].values()) - sum(self.entries["morale down"].values()) * self.morale_buff

        return atk, morale, lethal, self.entries["jinx"] > 0

    def on_defense(self):
        """

        :return: fragile        = double shield dmg
                 sleep, stun,   = ignore shield
                 morale, chill  = modify last stand entry
        """
        stun = False
        if self.entries["stun"]:
            self.entries["stun"] = False
            stun = True

        fragile = False
        if self.entries["fragile"]:
            self.entries["fragile"] = False
            fragile = True

        morale = (sum(self.entries["morale up"].values()) - sum(self.entries["morale down"])) * self.morale_buff

        return fragile, \
               self.entries["sleep"] > 0, \
               stun

    def on_dmg_calc(self):
        """
        :return: morale, chill  = modify last stand entry
        """
        return (sum(self.entries["morale up"].values()) - sum(self.entries["morale down"])) * self.morale_buff, \
               self.entries["chill"] > 0

    def on_tick(self):
        return self.entries["poison"] * self.poison_dmg

    def on_action(self):
        if self.entries["fear"]:
            self.entries["fear"] = max(self.entries["fear"]-1, 0)

    def on_turn(self):

        for entry in self.turns:
            self.entries[entry] = max(self.entries[entry]-1, 0)

        for entry in self.stack_turns:
            self.entries[entry] = {turns-1: stacks for turns, stacks in self.entries[entry].items() if turns-1 > 0}

    def to_str(self):
        return f'Buffs/Debuffs'
