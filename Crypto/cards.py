from Axie_models import Card, flatten

debuffs = ["aroma", "stench", "attack down", "morale down", "speed down",
           "fear", "chill", "stun", "poison", "jinx", "fragile", "lethal", "sleep"]


class Eggbomb(Card):

    def __init__(self):
        # card_name, body_part, part_name, element, cost, r, attack, defense, effect
        super().__init__("Eggbomb", "horn", "Eggshell", "bird", 1, "ranged", 120, 0,
                         "Apply aroma on this axie until next round")

    def on_attack(self, match, cards_played_this_turn, attacker, defender):
        self.owner.apply_stat_eff("aroma", turns=0)  # should disappear at start of next turn
        return 0


class DarkSwoop(Card):

    def __init__(self):
        super().__init__("Little owl", "mouth", "Dark swoop", "bird", 1, "melee", 25, 0,
                         "Target fastest enemy")

    def on_target_select(self, match, cards_played_this_turn, attacker, defender):
        final_speeds = {axie: 0 for axie in cards_played_this_turn.keys()}

        for axie, speed in [(axie, axie.speed) for axie in cards_played_this_turn.keys()]:

            speed_mult = 1

            # status effects
            for buff, count in axie.buffs.items():
                if buff == "speed up":
                    speed_mult += 0.2 * count
                if buff == "speed down":
                    speed_mult -= 0.2 * count

            final_speeds[axie] = speed * speed_mult

        # tODO only choose from opponents team

        return [],\
               [sorted([c for c in flatten(cards_played_this_turn.keys()) if c.player != self.owner.player],
                       key=lambda axie: final_speeds[axie], reverse=True)[0]]


class Blackmail(Card):

    def __init__(self):
        super().__init__("Pigeon post", "back", "Blackmail", "bird", 1, "ranged", 120, 10,
                         "Transfer all debuffs on this axie to target")

    def on_attack(self, match, cards_played_this_turn, attacker, defender):

        keys_to_del = []
        for debuff, (stacks, duration) in self.owner.buffs.items():
            if debuff in debuffs:
                defender.apply_stat_eff(debuff, stacks, duration)
                keys_to_del.append(debuff)
        for key in keys_to_del:
            del self.owner.buffs[key]

        return 0


class RiskyFeather(Card):

    def __init__(self):
        super().__init__("The last one", "tail", "Risky feather", "bird", 1, "ranged", 140, 10,
                         "Apply 2 Attack- to this axie")

    def on_attack(self, match, cards_played_this_turn, attacker, defender):
        self.owner.apply_stat_eff("attack down", times=2, turns=10000)  # goes away on attack
        return 0
