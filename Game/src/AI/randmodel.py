
import random


class RandomModel:
    def get_actions(self, states):
        actions = []
        for i in states:
            actions.append([random.randint(0, 1), random.randint(0, 1), random.randint(0, 1), random.randint(0, 1), random.randint(0, 1)])
            #actions.append([0, 0, 0, 0, 0])

        return actions, 0
    

    def store_rewards(self, rewards, dones):
        pass

    def memory_prep(self, num_players):
        pass

    def assign_device(self, device):
        pass
         

    