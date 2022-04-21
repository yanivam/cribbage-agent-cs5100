import random
from itertools import combinations
from random import choice, shuffle 

import numpy as np
import torch
from tqdm import tqdm
from collections import deque

from .score import score_hand, score_count
from .card import Deck

# Hyperparameters for RL agent:
from QModel import LinearQNet, QTrainer
MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001


class WinGame(Exception):
    pass


class Player:
    """
    Here we define a base class for the various kinds of players in the 
    cribbage game. To define your own player, inherit from this class and 
    implement ``ask_for_input`` and ``ask_for_discards``
    """

    def __init__(self, name=None):
        self.name = name
        self.hand = []  # cards in player's hand
        self.table = [] # cards on table in front of player
        self.crib = []  # cards in player's crib
        self.score = 0

    # Discards

    def ask_for_discards(dealer=0):
        """Should return two cards from the player"""
        raise Exception("You need to implement `ask_for_discards` yourself")


    def update_after_discards(self, discards):
        for discard in discards:
            self.hand.remove(discard)


    def discards(self, dealer=0):
        cards = self.ask_for_discards(dealer)
        self.update_after_discards(cards)
        return cards


    # Counting plays

    def ask_for_play(self, previous_plays, turn=0):
        """Should return a single card from the player

        Private method"""
        raise Exception("You need to implement `ask_for_play` yourself")


    def update_after_play(self, play):
        """Private method"""
        self.table.append(play)
        self.hand.remove(play)


    def play(self, count, previous_plays, turn=0):
        """Public method"""
        if not self.hand:
            print('>>> I have no cards', self)
            return "No cards!"
        elif all(count + card.value > 31 for card in self.hand):
            print(">>>", self, self.hand, "I have to say 'go' on that one")
            return "Go!"
        while True:
            card = self.ask_for_play(previous_plays, turn)  # subclasses (that is, actual players) must implement this
            #print("Nominated card", card)
            if sum((pp.value for pp in previous_plays)) + card.value < 32:
                self.update_after_play(card)
                return card
            else: 
                # `self.ask_for_play` has returned a card that would put the 
                # score above 31 but the player's hand contains at least one
                # card that could be legally played (you're not allowed to say
                # "go" here if you can legally play). How the code knows that 
                # the player has a legal move is beyond me
                print('>>> You nominated', card, 'but that is not a legal play given your hand. You must play if you can')

    def play_step(self, count, previous_plays, turn):
        pass

    # Scoring
    def peg(self, points):
        self.score += points
        if self.score > 120:
            self.win_game()


    def count_hand(self, turn_card):
        """Count the hand (which should be on the table)"""
        score = score_hand(self.table, turn_card)
        self.peg(score)
        return score 


    def count_crib(self, turn_card):
        """Count crib, with side effect of pegging the score"""
        score = score_hand(self.crib, turn_card)
        self.peg(score)
        return score 


    def win_game(self):
        raise WinGame(f"""Game was won by {self}  """
                      f"""{self.score} {self.table}""")


    @property
    def sorted_hand(self):
        return sorted(self.hand, key=lambda c: c.index)

    def __repr__(self):
        if self.name:
            return self.name + f'(score={self.score})'
        return str(self.__class__) + f'(score={self.score})'


class RandomPlayer(Player):
    """
    A player who plays randomly
    """

    def ask_for_play(self, previous_plays, turn=0):
        shuffle(self.hand) # this handles a case when 0 is not a legal play
        return self.hand[0]


    def ask_for_discards(self, dealer=0):
        # and isn't needed here 
        return self.hand[0:2]


class HumanPlayer(Player):
    """
    A human player. This class implements scene rendering and taking input from
    the command line 
    """


    def ask_for_play(self, previous_plays, turn=0):
        """Ask a human for a card during counting"""
        
        d = dict(enumerate(self.hand, 1))
        print(f">>> Your hand ({self}):", " ".join([str(c) for c in self.hand]))

        while True:
            inp = input(">>> Card number to play: ") or "1"
            if len(inp) > 0 and int(inp) in d.keys():
                card = d[int(inp)]
                return card


    def ask_for_discards(self, dealer=0):
        """After deal, ask a human for two cards to discard to crib"""

        d = dict(enumerate(self.sorted_hand, 1))

        print('>>> Please nominate two cards to discard to the crib')
        print(f'>>> {d[1]} {d[2]} {d[3]} {d[4]} {d[5]} {d[6]}')
        discard_prompt = ">>> "

        while True:
            inp = input(discard_prompt) or "12"
            cards = [d[int(i)] for i in inp.replace(" ", "").replace(",", "")]
            if len(cards) == 2:
                print(f">>> Discarded {cards[0]} {cards[1]}")
                return cards


class RLAgent(Player):
    """
    A player that uses a RL trained model to select cards to play. Other
    """
    # Init should load the model and set it to eval mode
    def __init__(self):
        super().__init__()
        self.discarded = []  # Will represent the discards that the player *knows* it discarded to crib
        self.n_games = 0
        self.epsilon = 0
        self.gamma = 0.9
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = LinearQNet(209, 40, 6)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def ask_for_play(self, previous_plays, turn=0):
        # Convert hand and previous_plays into an input to the NN
        # pass this input through the model
        # convert model output into card choice
        pass

    def ask_for_discards(dealer=0):
        # Convert hand and previous plays into an input to th NN
        # pass this input through the model
        # convert model output into card choice
        pass

    def _get_state(self, previous_plays, dealer):
        '''
        Get the state of play as a one-hot encoded np.array.

        One input should represent whether or not the player is the dealer.
        52 inputs represent 1-hot vector of whether or not card is in hand.
        52 inputs represent "known" cards - i.e. the cards in the crib
        104 inputs represent the state of play - the first 13 represent the rank of the first card
        played, the second 13 represent the rank of the second, and so on.
        '''
        state = np.zeros(209)
        if dealer == 1:
            state[0] = 1

        # Set state of cards in hand:
        for card in self.hand:
            idx = self._get_card_index(card)
            idx = idx + 1  # Add one as we've already set the dealer state
            state[idx] = 1

        # Set state of "known cards" that were discarded to the crib:
        for card in self.discarded:
            idx = self._get_card_index(card)
            idx = idx + 52 + 1
            state[idx] = 1

        # Set the state of play:
        for i, card in enumerate(previous_plays):
            card_idx = self._get_card_index(card)
            idx = card_idx + (i * 13) + 52 + 52 + 1
            state[idx] = 1

        return state

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        self.epsilon = 80 - self.n_games
        final_move = 0
        if random.randint(0, 200) < self.epsilon:
            final_move = random.randint(0, 5)
        else:
            state0 = torch.Tensor(state)
            prediction = self.model(state0)
            final_move = torch.argmax(prediction).item()
        return final_move

    def _get_card_index(self, card):
        suit = card.suit
        value = card.rank
        return suit * 13 + value


class GreedyAgentPlayer(Player):
    """
    "Expert systems" style AI player that systematically
    enumerates possible moves and chooses the move that
    maximizes its score after the move
    """

    def ask_for_discards(self, dealer=0):
        """
        For each possible discard, score and select
        highest scoring move. Note: this will give opponents 
        excellent cribs, needs a flag for minimizing 
        """

        print("cribbage: {} is choosing discards".format(self))
        deck = Deck().draw(52)
        potential_cards = [n for n in deck if n not in self.hand]
        bar = tqdm(total=226994)
        discards = []
        mean_scores = []
        for discard in combinations(self.hand, 2):  # 6 choose 2 == 15
            inner_scores = []
            for pot in combinations(potential_cards, 3):  # 46 choose 3 == 15,180
                inner_scores.append(score_hand([*discard, *pot[:-1]], pot[-1]))
                bar.update(1)
            inner_scores = np.array(inner_scores)
            discards.append(discard)
            mean_scores.append(inner_scores.mean())

        return list(discards[np.argmin(mean_scores)])


    def ask_for_play(self, previous_plays, turn=0):
        """
        Calculate points for each possible play in your hand
        and choose the one that maximizes the points
        """

        scores = []
        plays = []
        for card in self.hand:
            plays.append(card)
            scores.append(score_count(previous_plays + [card]))
        max_index = np.argmax(scores)

        return plays[max_index]


# class TrainedAIPlayer(Player):
#     """
#     A player that makes choices based on previous games
#     """

#     def __init__(self, name=""):
#         # override this constructor becasue I want to
#         # load the ML model in when we init AIPlayer instance
#         self.name = name
#         self.hand = []
#         self.score = 0
#         self.debug = False
#         self.model = load_trained_model()  # trained model we can ask directly for plays

#     def ask_for_input(self, play_vector):
#         card = self.model.ask_for_pegging_play(play_vector, self.in_hand)
#         card.ontable = True
#         return card

#     def ask_for_discards(self):
#         cards = self.model.ask_model_for_discard(
#             self.hand
#         )  # note: returns card objects
#         self.hand = [n for n in self.hand if n not in cards]
#         return cards


NondeterministicAIPlayer = RandomPlayer
