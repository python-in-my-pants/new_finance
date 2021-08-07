from Axie_models import *
from cards import *

p1 = Player([Axie("bird", DarkSwoop(), Eggbomb(), Blackmail(), RiskyFeather(), "bird", "bird"),
             Axie("bird", DarkSwoop(), Eggbomb(), Blackmail(), RiskyFeather(), "bird", "bird"),
             Axie("bird", DarkSwoop(), Eggbomb(), Blackmail(), RiskyFeather(), "bird", "bird")])

p2 = Player([Axie("bird", DarkSwoop(), Eggbomb(), Blackmail(), RiskyFeather(), "bird", "bird"),
             Axie("bird", DarkSwoop(), Eggbomb(), Blackmail(), RiskyFeather(), "bird", "bird"),
             Axie("bird", DarkSwoop(), Eggbomb(), Blackmail(), RiskyFeather(), "bird", "bird")])

battle = Match(p1, p2)
battle.run_simulation()
