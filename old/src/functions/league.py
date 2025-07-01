from AI.randmodel import RandomModel
from functions.ELO import ELO
from random import randint
from params import AIHyperparameters

class League:

    def __init__(self, elo_env, opponent):
        AI_PARAMS = AIHyperparameters()

        self.players = []
        self.elos = []
        self.num_players = 0
        self.max_players = AI_PARAMS.max_oppenents
        self.elo_env = elo_env
        self.age = 0 
        self.max_age = AI_PARAMS.opposing_model_freeze_time   
        self.opponent = opponent

        

    def add_player(self, player, elo):
        self.players.append(player.state_dict())
        self.elos.append(self.elo_env.create_rating(elo.mu, elo.sigma))
        self.num_players += 1

    def remove_player(self):
        #remove first player on list
        self.players.pop(0)
        self.elos.pop(0)
        self.num_players -= 1

    def sample_player(self):
        if randint(0,100) < 10:
            return RandomModel(), None
        else:
            index = randint(0, self.num_players-1)
            self.opponent.load_state_dict(self.players[index])
            return self.opponent, self.elos[index]

    def update(self, player, elo):
        self.age += 1
        if self.age > self.max_age:
            self.add_player(player, elo)
            self.age = 0
            if self.num_players > self.max_players:
                self.remove_player()
        