import json
from params import EnvironmentHyperparameters


class Logger:
    def __init__(self, log_name):
        self.log_name = log_name
        self.file = open(self.log_name, 'w')
        self.data = {
            'parameters': {},
            'states': []
        }

    def write_parameters(self, env_params):
        # Convert env_params to a dictionary
        params_dict = env_params.__dict__
        self.data['parameters'] = params_dict

    def log_state(self, players, ball, timer, score):
        goals1, goals2 = score
        state = {
            'time': timer,
            'players': [],
            'ball': {
                'position': ball.position.copy(),
                'velocity': ball.velocity.copy()
            },
            'score' : [goals1, goals2]
        }
        for player in players:
            player_state = {
                'team_id': player.team_id,
                'player_id': player.player_id,
                'position': player.position.copy(),
                'velocity': player.velocity.copy()
            }
            state['players'].append(player_state)
        self.data['states'].append(state)

    def close(self):
        # Write data to file in JSON format
        json.dump(self.data, self.file)
        self.file.close()


def set_parameters(parameters):
    Env_params = EnvironmentHyperparameters()
    CAP_FPS = Env_params.CAP_FPS
    for key, value in parameters.items():
        if hasattr(Env_params, key):
            setattr(Env_params, key, value)

    setattr(Env_params, "RENDER", True)
    setattr(Env_params, "CAP_FPS", CAP_FPS)
    print(f"CAP_FPS: {CAP_FPS}")

