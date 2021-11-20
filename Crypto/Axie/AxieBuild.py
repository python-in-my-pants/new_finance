from Axie_models import Axie
from DDict import DDict


class AxieBuild:

    """
    container for axie builds

    can be created from vague descriptions like
    180 shield and min 150 beast dmg with 1 backdoor and 1 zero cost
    """

    def __init__(self,
                 elements=None,
                 horn=[],
                 tail=[],
                 mouth=[],
                 back=[],

                 min_hp=0,
                 min_speed=0,
                 min_skill=0,
                 min_morale=0,

                 min_shield=0,
                 min_avg_shield=0,
                 min_med_shield=0,

                 min_beast_dmg=0,
                 min_aqua_dmg=0,
                 min_plant_dmg=0,
                 min_cum_dmg=0,
                 min_med_dmg=0,

                 tags=[],
                 cum_max_cost=10,
                 cum_min_cost=0,
                 min_zero_cost=0
                 ):

        self.back = back
        self.mouth = mouth
        self.tail = tail
        self.horn = horn
        self.elements = elements

        self.min_hp = min_hp
        self.min_speed = min_speed
        self.min_skill = min_skill
        self.min_morale = min_morale

        self.min_shield = min_shield
        self.min_avg_shield = min_avg_shield
        self.min_med_shield = min_med_shield

        self.min_beast_dmg = min_beast_dmg
        self.min_aqua_dmg = min_aqua_dmg
        self.min_plant_dmg = min_plant_dmg
        self.min_cum_dmg = min_cum_dmg
        self.min_med_dmg = min_med_dmg

        self.tags = tags
        self.cum_max_cost = cum_max_cost
        self.cum_min_cost = cum_min_cost
        self.min_zero_cost = min_zero_cost

        self.parts = {
            "element": [],
            "mouth": [],
            "horn": [],
            "back": [],
            "tail": [],
        }

    def compliant(self, axie_id, omit_parts=()):
        """

        :param axie_id:
        :param omit_parts: body parts not to check for
        :return:
        """
        model = Axie.from_id(axie_id)
        ...  # TODO
        return True


builds = DDict({
    "backdoor_bird": {
        "classes": ["bird"],
        "parts": {
            "back": ["pigeon post", "balloon"],
            "mouth": ["little owl"],
            "tail": ["hare", "post fight", "the last one", "cloud"],
            "horn": ["eggshell"]
        },
        "genes": {"printer": 1},
        "speed": [61, 61]
    },
    "double_anemone": {
        "classes": ["aqua", "plant", "reptile", "dusk"],
        "parts": {
            "back": ["anemone"],
            "mouth": ["catfish", "zigzag", "mosquito"],
            "tail": ["koi", "post fight", "nimo", "snake jar", "wall gecko"],
            "horn": ["anemone"]
        },
        "genes": {"printer": 1},
    },
    "damage_reflector": {
        "classes": ["reptile"],
        "parts": {
            "back": ["indian star"],
            "mouth": ["tiny turtle", "kotaro"],
            "tail": ["wall gecko", "snake jar"],
            "horn": ["scaly spoon"]
        },
        "hp": [50, 61],
        "genes": {"printer": 2},
    },

    "control_beast": {
        "classes": ["beast"],
        "parts": {
            "back": ["risky beast", "jaguar"],
            "mouth": ["goda"],
            "tail": ["rice", "cottontail"],
            "horn": ["dual blade", "pliers", "pocky", "little branch"]
        },
        "speed": [43, 61],
        "genes": {"printer": 2},
    },
    "damage_beast": {
        "classes": ["beast"],
        "parts": {
            "back": ["risky beast", "jaguar"],
            "mouth": ["nut cracker"],
            "tail": ["nut throw"],
            "horn": ["dual blade", "pliers", "pocky", "little branch"]
        },
        "speed": [43, 61],
        "genes": {"printer": 2},
    },
    "rimp_beast": {
        "classes": ["beast"],
        "parts": {
            "back": ["ronin"],
            "mouth": ["nut cracker"],
            "tail": ["nut throw"],
            "horn": ["imp"]
        },
        "speed": [43, 61],
        "genes": {"printer": 2},
    },
    "rimp_control_beast": {
        "classes": ["beast"],
        "parts": {
            "back": ["ronin"],
            "mouth": ["goda"],
            "tail": ["rice", "cottontail"],
            "horn": ["imp"]
        },
        "speed": [43, 61],
        "genes": {"printer": 2},
    },

    "terminator": {
        "classes": ["reptile", "dusk"],
        "parts": {
            "back": ["snail shell"],
            "mouth": ["tiny turtle"],
            "tail": ["thorny caterpillar"],
            "horn": ["lagging"]
        },
        "speed": [43, 61],
        "genes": {"printer": 2},
    },
    "dusk_terminator": {
        "classes": ["dusk"],
        "parts": {
            "back": ["snail shell"],
            "mouth": ["tiny turtle"],
            "tail": ["thorny caterpillar"],
            "horn": ["lagging"]
        },
        "speed": [43, 61],
        "genes": {"printer": 2},
    },
    "max_speed_dusk_terminator": {
        "classes": ["dusk"],
        "parts": {
            "back": ["snail shell"],
            "mouth": ["tiny turtle"],
            "tail": ["thorny caterpillar"],
            "horn": ["lagging"]
        },
        "speed": [46, 61],
        "genes": {"printer": 2},
    },

    "kamikaze_bird": {
        "classes": ["bird"],
        "parts": {
            "back": ["perch", "tri spikes"],
            "mouth": ["axie kiss"],
            "tail": ["post fight"],
            "horn": ["dual blade"]
        },
        "genes": {"printer": 1},
        "speed": [59, 61]
    },
    "kamikaze_beast": {
        "parts": {
            "back": ["perch", "ronin", "tri spikes", "risky beast", "jaguar", "pigeon post"],
            "mouth": ["axie kiss"],
            "tail": ["post fight"],
            "horn": ["dual blade"]
        },
        "genes": {"printer": 1},
        "speed": [43, 61]
    },
})
