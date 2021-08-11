from Axie.Axie_models import *
from Axie.cards import *

"""
p1 = Player([Axie("bird", DarkSwoop(), Eggbomb(), Blackmail(), RiskyFeather()),
             Axie("bird", DarkSwoop(), Eggbomb(), Blackmail(), RiskyFeather()),
             Axie("bird", DarkSwoop(), Eggbomb(), Blackmail(), RiskyFeather())])

p2 = Player([Axie("bird", DarkSwoop(), Eggbomb(), Blackmail(), RiskyFeather()),
             Axie("bird", DarkSwoop(), Eggbomb(), Blackmail(), RiskyFeather()),
             Axie("bird", DarkSwoop(), Eggbomb(), Blackmail(), RiskyFeather())])
"""

p1, p2 = Player.get_random(), Player.get_random()

battle = Match(p1, p2)
battle.run_simulation()
