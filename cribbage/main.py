from argparse import ArgumentParser

import random
from .game import Game
from .players import (
    WinGame,
    HumanPlayer,
    NondeterministicAIPlayer,
    GreedyAgentPlayer,
    RLAgent,
    RandomPlayer,
    HeuristicAgentPlayer
)
import multiprocessing
import time

def playCribbageOnce(i):
    print(f"start process {i}")
    randomPlayerTurn = [HeuristicAgentPlayer('Player 1'), GreedyAgentPlayer('Player 2')]
    random.shuffle(randomPlayerTurn)
    game = Game(randomPlayerTurn[0], randomPlayerTurn[1])
    try:
        game.run()
    except WinGame as win_game:
        print(f"end process {i}")
        return win_game

def play1000CribbageGames():
    pool = multiprocessing.Pool(10)
    winners = pool.map(playCribbageOnce, range(11))
    pool.close()
    pool.join()
    return winners

def main():
    # Take inputs from command line
    player_choices = {
        "human": HumanPlayer,
        "greedy": GreedyAgentPlayer,
        "heuristic": HeuristicAgentPlayer,
        "random": NondeterministicAIPlayer,
    }
    '''
    parser = ArgumentParser()
    parser.add_argument("player1", help="Type of player 1", choices=player_choices)
    parser.add_argument("player2", help="Type of player 2", choices=player_choices)
    parser.add_argument("-a", "--name_1", help="Optional name for player 1")
    parser.add_argument("-b", "--name_2", help="Optional name for player 2")
    args = parser.parse_args()
    '''

    # Play game
    print("", multiprocessing.cpu_count() - 1)
    start_time = time.time()
    winners = play1000CribbageGames()
    duration = time.time() - start_time
    print('Played 1000 games in ' + str(duration) + ' seconds')

    with open("test.txt", 'w', encoding='utf-8') as f:
        p_sum = sum(1 if 'Game was won by Player 1' in str(winner) else 0 for winner in winners) / 11
        for winner in winners:
            if 'Game was won by Player 1' in str(winner):
                f.write(str(1))
            else:
                f.write(str(0))
    print('Percentage won by greedy agent: ' + str(p_sum))