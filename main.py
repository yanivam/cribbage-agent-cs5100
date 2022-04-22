from argparse import ArgumentParser
import sys
import os

import numpy as np
from tqdm import tqdm
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from contextlib import redirect_stdout


from cribbage.game import Game
from cribbage.players import (
    WinGame,
    HumanPlayer,
    NondeterministicAIPlayer,
    EnumerativeAIPlayer,
    TreeAIPlayer,
)

if __name__ == "__main__":

    # Take inputs from command line
    player_choices = {
        "human": HumanPlayer,
        "expert": EnumerativeAIPlayer,
        "random": NondeterministicAIPlayer,
        "tree":TreeAIPlayer,
    }
    parser = ArgumentParser()
    parser.add_argument("test_against", help="Run 1000 trails to evaluate agents against each other", choices={'True':True, 'False':False})
    parser.add_argument("player1", help="Type of player 1", choices=player_choices)
    parser.add_argument("player2", help="Type of player 2", choices=player_choices)
    parser.add_argument("-a", "--name_1", help="Optional name for player 1")
    parser.add_argument("-b", "--name_2", help="Optional name for player 2")
    args = parser.parse_args()

    
    

    run_time = 1
    if args.test_against == 'True':
        run_time = 1000
    game_records = []
    bar = tqdm(total=run_time)

    for i in range(run_time):
        # Set up players
        player_1 = player_choices[args.player1](args.player1)#'Player 1')#name=args.name1 or 
        player_2 = player_choices[args.player2](args.player2)#'Player 2')#name=args.name2 or
        # Play game
        game = Game(player_1, player_2)
        try:
            if args.test_against == 'True':
                with open('out.txt', 'w') as f:
                    with redirect_stdout(f):
                        game.run()
            else:
                game.run()
        except WinGame as win_game:
            print(win_game)
        s1,s2 = player_1.score, player_2.score
        winner = 1 if s1>=s2 else 0
        game_records.append([winner,s1,s2])
            
        bar.update(1)
    game_records_T = np.transpose(game_records)
    df = pd.DataFrame(game_records,columns = ['Winner','Score1','Score2'])

    plt.style.use('seaborn-deep')
    bins = np.linspace(80, 130, 50)
    plt.figtext(0.15, 0.9, f"Player 1 has a {np.mean(game_records_T[0])} winning rate", fontsize=12)
    plt.hist([game_records_T[1],game_records_T[2]], label=[f"Player1 ({args.player1}) scores",f"Player2 ({args.player2}) scores"])  
    plt.legend(loc='upper left')#, numpoints=1, bbox_to_anchor=(0.5, -0.05), ncol=2, fancybox=True, shadow=True)
    

    plt.savefig('data.png')  
    # plt.show() 
    # plt.close()



