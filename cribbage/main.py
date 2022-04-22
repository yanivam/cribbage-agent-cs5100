from argparse import ArgumentParser
import sys
import os
print("Hey is anyone here")
# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# # print(SCRIPT_DIR)
# sys.path.append(os.path.dirname(SCRIPT_DIR))
from cribbage.game import Game
from cribbage.players import (
    WinGame,
    HumanPlayer,
    NondeterministicAIPlayer,
    EnumerativeAIPlayer,
    TreeAIPlayer,
)


def main():

    # Take inputs from command line
    player_choices = {
        "human": HumanPlayer,
        "expert": EnumerativeAIPlayer,
        "random": NondeterministicAIPlayer,
        "tree":TreeAIPlayer,
    }
    print("Hey is anyone here")

    parser = ArgumentParser()
    parser.add_argument("player1", help="Type of player 1", choices=player_choices)
    parser.add_argument("player2", help="Type of player 2", choices=player_choices)
    parser.add_argument("-a", "--name_1", help="Optional name for player 1")
    parser.add_argument("-b", "--name_2", help="Optional name for player 2")
    args = parser.parse_args()

    # Set up players
    player_1 = player_choices[args.player1](name=args.name1 or 'Player 1')
    player_2 = player_choices[args.player2](name=args.name2 or 'Player 2')

    # Play game
    game = Game(player_1, player_2)
    try:
        game.run()
    except WinGame as win_game:
        print(win_game)
