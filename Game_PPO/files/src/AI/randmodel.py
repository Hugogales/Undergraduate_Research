
import random


class RandomModel:
    def move(self, inputs):
        #return [0,0,1,0] # Move left
        return [random.randint(0, 1), random.randint(0, 1), random.randint(0, 1), random.randint(0, 1)]

    