from enviroment.Game import Game
import multiprocessing
import time
import pygame

pygame.init()

def run_game():
    game = Game()
    game.run()

if __name__ == "__main__":
    start_time = time.time()
    
    processes = []

    # Start Game 1 with rendering
    for i in range(1):
        p1 = multiprocessing.Process(target=run_game)
        processes.append(p1)
        p1.start()

    for p in processes:
        p.join()

    print("Total time taken: ", time.time() - start_time)
