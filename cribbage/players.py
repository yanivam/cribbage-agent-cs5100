from itertools import combinations
from random import shuffle

import os
import sys

import numpy as np
import pkg_resources
import torch

from .score import score_hand, score_count
from .card import Deck, Card


from .score import score_hand, score_count, score_hand_heuristic
from .card import Deck

from .minimax import minimaxTree

# Model for RL agent:
from .QModel import LinearQNet

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

    def ask_for_discards(self, dealer=0):
        """Should return two cards from the player"""
        raise Exception("You need to implement `ask_for_discards` yourself")

    def update_after_discards(self, discards):
        for discard in discards:
            self.hand.remove(discard)

    def discards(self, dealer=0):
        cards = self.ask_for_discards(dealer=dealer)
        self.update_after_discards(cards)
        return cards

    # Counting plays
    def ask_for_play(self,previous_plays, turn=0, count = 0):
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
            #print('>>> I have no cards', self)
            return "No cards!"
        elif all(count + card.value > 31 for card in self.hand):
            #print(">>>", self, self.hand, "I have to say 'go' on that one")
            return "Go!"
        while True:
            card = self.ask_for_play(previous_plays, turn)  # subclasses (that is, actual players) must implement this
            #print("Nominated card", card)
            if sum((pp.value for pp in previous_plays)) + card.value < 32:
                self.update_after_play(card)
                return card
            #else:
                # `self.ask_for_play` has returned a card that would put the 
                # score above 31 but the player's hand contains at least one
                # card that could be legally played (you're not allowed to say
                # "go" here if you can legally play). How the code knows that 
                # the player has a legal move is beyond me
                #print('>>> You nominated', card, 'but that is not a legal play given your hand. You must play if you can')

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

    def ask_for_play(self, previous_plays, turn=0,count = 0):
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


    def ask_for_play(self, previous_plays, turn=0,count = 0):
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
    def __init__(self, name=None):
        super().__init__(name)
        self.playable_cards = []
        self.discarded = []  # Will represent the discards that the player *knows* it discarded to crib
        self.discard_model = LinearQNet(209, 60, 52)
        self.pegging_model = LinearQNet(209, 60, 52)
        discard_path = pkg_resources.resource_filename('cribbage', 'simple_model.pth')
        peg_path = pkg_resources.resource_filename('cribbage', 'simple_pegging_model.pth')
        self.discard_model.load_state_dict(torch.load(discard_path))
        self.pegging_model.load_state_dict(torch.load(peg_path))

    def play(self, count, previous_plays, turn=0):
        '''
        Overrides parent play method to pass the current count along to self.ask_for_play()
        '''
        if not self.hand:
            print('>>> I have no cards', self)
            return "No cards!"
        elif all(count + card.value > 31 for card in self.hand):
            print(">>>", self, self.hand, "I have to say 'go' on that one")
            return "Go!"

        # Only ask for a play if we have cards we can actually play
        card = self.ask_for_play(previous_plays, turn,count)
        self.update_after_play(card)
        return card


    def ask_for_play(self, previous_plays, dealer=0, count = 0):
        # Limit ourselves to cards we can actually play:
        self.playable_cards = [c for c in self.hand if c.value + count <= 31]
        state = torch.Tensor(self.get_state(self.hand, previous_plays, dealer))

        output = self.pegging_model(state)

        vals, indices = output.sort(descending=True)

        for idx in indices:
            card = Card(idx)
            if card in self.hand and card in self.playable_cards:
                return card


    def _discard(self, dealer):
        '''
        Private helper method for performing a single discard
        '''
        # Get state appropriate for NN
        state = torch.Tensor(self.get_state(self.hand, [], dealer))

        # Pass input through discard model:
        output = self.discard_model(state)
        vals, indices = output.sort(descending=True)
        for idx in indices:
            card = Card(idx)

            if card in self.hand and card not in self.discarded:
                self.discarded.append(card)
                return card

    def ask_for_discards(self, dealer=0):
        self.discarded = []  # Reset the discarded cards
        card1 = self._discard(dealer)
        card2 = self._discard(dealer)
        return [card1, card2]

    def get_state(self, in_hand, previous_plays, dealer):
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
        for card in in_hand:
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
            card_idx = card.rank
            idx = card_idx + (i * 13) + 52 + 52 + 1
            state[idx] = 1

        return state

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
        For each possible discard, score and select highest scoring move.

        This operates by enumerating every possible combination of turn card and crib and using them to compute an
        expected crib value (if non-dealer) or hand + crib value (if dealer) for each possible discard. If the agent is
        the dealer, it chooses the discard with the highest expected hand + crib score. If it is not the dealer, the
        agent chooses the discard that minimizes the crib value.
        """

        #print("cribbage: {} is choosing discards".format(self))
        deck = Deck().draw(52)
        potential_cards = [n for n in deck if n not in self.hand]
        discards = []
        mean_scores = []
        for discard in combinations(self.hand, 2):  # 6 choose 2 == 15
            inner_scores = []  # Keep track of the Expected score for each discard
            hand_after_discard = [c for c in self.hand if c not in discard]  # The cards that remain after discarding
            for pot in combinations(potential_cards, 3):  # 46 choose 3 == 15,180
                if not dealer:
                    # Our goal is to minimize the crib, so we only compute crib scores:
                    inner_scores.append(score_hand([*discard, *pot[:-1]], pot[-1]))
                else:
                    # Score of the cards remaining in your hand:
                    score_of_remaining_cards = score_hand(hand_after_discard, pot[-1])
                    # Update inner scores with the sum of the score for remaining cards plus the expected score
                    # of the crib (because the dealer takes both the crib and its hand):
                    inner_scores.append(score_hand([*discard, *pot[:-1]], pot[-1]) + score_of_remaining_cards)
            inner_scores = np.array(inner_scores)
            discards.append(discard)
            mean_scores.append(inner_scores.mean())

        if dealer:
            # If we're the dealer, we want to maximize our expected score:
            return list(discards[np.argmax(mean_scores)])
        else:
            # Otherwise, minimize the expected crib score:
            return list(discards[np.argmin(mean_scores)])
        # return self.hand[0:2]


    def ask_for_play(self,  previous_plays, turn=0, count = 0):
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

class HeuristicAgentPlayer(Player):
    """
    "Expert systems" style AI player that systematically
    enumerates possible moves and chooses the move that
    maximizes its score after the move
    """

    def expertStrategyHeuristic(self, discards, scores, dealer):
        new_discards = discards.copy()
        new_scores = scores.copy()
        if not dealer:
            for discard in discards:
                if(sum(list(map(lambda d: d.value, discard))) == 5):
                    remIdx = new_discards.index(discard)
                    new_discards.pop(remIdx)
                    new_scores.pop(remIdx)
                elif("J" in list(map(lambda d: d.rank_str, discard)) or "Q" in list(map(lambda d: d.rank_str, discard)) or "3" in list(map(lambda d: d.rank_str, discard)) or "4" in list(map(lambda d: d.rank_str, discard)) or "7" in list(map(lambda d: d.rank_str, discard)) or "8" in list(map(lambda d: d.rank_str, discard))):
                    remIdx = new_discards.index(discard)
                    new_discards.pop(remIdx)
                    new_scores.pop(remIdx)
                elif (abs(discard[0].value - discard[1].value) == 2):
                    remIdx = new_discards.index(discard)
                    new_discards.pop(remIdx)
                    new_scores.pop(remIdx)
                elif (discard[0].suit == discard[1].suit):
                    remIdx = new_discards.index(discard)
                    new_discards.pop(remIdx)
                    new_scores.pop(remIdx)
        else:
            for discard in discards:
                if(not sum(list(map(lambda d: d.value, discard))) == 5):
                    remIdx = new_discards.index(discard)
                    new_discards.pop(remIdx)
                    new_scores.pop(remIdx)
                    continue
                if ("J" in list(map(lambda d: d.rank_str, discard)) or "Q" in list(map(lambda d: d.rank_str, discard)) or "3" in list(map(lambda d: d.rank_str, discard)) or "4" in list(map(lambda d: d.rank_str, discard)) or "7" in list(map(lambda d: d.rank_str, discard)) or "8" in list(map(lambda d: d.rank_str, discard))):
                    remIdx = new_discards.index(discard)
                    new_discards.pop(remIdx)
                    new_scores.pop(remIdx)
                    continue
                if (abs(discard[0].value - discard[1].value) == 2):
                    remIdx = new_discards.index(discard)
                    new_discards.pop(remIdx)
                    new_scores.pop(remIdx)
                    continue
                if (discard[0].suit == discard[1].suit):
                    remIdx = new_discards.index(discard)
                    new_discards.pop(remIdx)
                    new_scores.pop(remIdx)
                    continue
        return new_discards, new_scores
    
    def ask_for_discards(self, dealer=0):
        """
        For each possible discard, score and select
        highest scoring move. Note: this will give opponents 
        excellent cribs, needs a flag for minimizing 
        """

        #print("cribbage: {} is choosing discards".format(self))
        deck = Deck().draw(52)
        discards = []
        scores = []
        for discard in combinations(self.hand, 2):  # 6 choose 2 == 15
            discards.append(discard)
            scores.append(0)
        discards, scores = self.expertStrategyHeuristic(discards, scores, dealer)
        for discardIdx in range(len(discards)):
            scores[discardIdx] = score_hand_heuristic([x for x in self.hand if x not in discards[discardIdx]])
        if(len(scores) == 0):
            for discard in combinations(self.hand, 2):  # 6 choose 2 == 15
                discards.append(discard)
                scores.append(0)
            for discardIdx in range(len(discards)):
                scores[discardIdx] = score_hand_heuristic([x for x in self.hand if x not in discards[discardIdx]])
        return list(discards[np.argmin(scores)])


    def ask_for_play(self, previous_plays,turn = 0, count = 0):
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

class TreeAIPlayer(Player):

    def ask_for_discards(self,dealer=0):
        """
        For each possible discard, score and select
        highest scoring move. 
        """

        # print("cribbage: {} is choosing discards".format(self))
        # deck = Deck().draw(52)
        # potential_cards = [n for n in deck if n not in self.hand]
        # bar = tqdm(total=226994)
        # discards = []
        # mean_scores = []
        # for discard in combinations(self.hand, 2):  # 6 choose 2 == 15
        #     inner_scores = []
        #     for pot in combinations(potential_cards, 3):  # 46 choose 3 == 15,180
        #         inner_scores.append(score_hand([*discard, *pot[:-1]], pot[-1]))
        #         bar.update(1)
        #     inner_scores = np.array(inner_scores)
        #     discards.append(discard)
        #     mean_scores.append(inner_scores.mean())

        # # return either the best (if my crib) or the worst (if not)
        # if my_crib:
        #     selected = np.argmax(mean_scores)
        # else:
        #     selected = np.argmin(mean_scores)

        # return list(discards[selected])




        # deck = Deck().draw(52)
        # potential_cards = [n for n in deck if n not in self.hand]
        # discards = []
        # mean_scores = []
        # for discard in combinations(self.hand, 2):  # 6 choose 2 == 15
        #     inner_scores = []  # Keep track of the Expected score for each discard
        #     hand_after_discard = [c for c in self.hand if c not in discard]  # The cards that remain after discarding
        #     for pot in combinations(potential_cards, 3):  # 46 choose 3 == 15,180
        #         if not dealer:
        #             # Our goal is to minimize the crib, so we only compute crib scores:
        #             inner_scores.append(score_hand([*discard, *pot[:-1]], pot[-1]))
        #         else:
        #             # Score of the cards remaining in your hand:
        #             score_of_remaining_cards = score_hand(hand_after_discard, pot[-1])
        #             # Update inner scores with the sum of the score for remaining cards plus the expected score
        #             # of the crib (because the dealer takes both the crib and its hand):
        #             inner_scores.append(score_hand([*discard, *pot[:-1]], pot[-1]) + score_of_remaining_cards)
        #     inner_scores = np.array(inner_scores)
        #     discards.append(discard)
        #     mean_scores.append(inner_scores.mean())

        # if dealer:
        #     # If we're the dealer, we want to maximize our expected score:
        #     return list(discards[np.argmax(mean_scores)])
        # else:
        #     # Otherwise, minimize the expected crib score:
        #     return list(discards[np.argmin(mean_scores)])




        return self.hand[0:2]



    def ask_for_play(self, hand, sequence, current_sum):
        """
        Generate a tree that are used to maximize the expected utility and to recommend next play
        hand: current hand
        sequence: previous cards in this play
        current_sum: running total of cards in this play
        """
        tree = minimaxTree(hand, sequence, current_sum, 3)
        return tree.recommendCard(0)

    def play(self, count, previous_plays, turn=0):
        """Public method"""
        if not self.hand:
            print('>>> I have no cards', self)
            return "No cards!"
        elif all(count + card.value > 31 for card in self.hand):
            print(">>>", self, self.hand, "I have to say 'go' on that one")
            return "Go!"
        while True:
            card = self.ask_for_play(self.hand, previous_plays,count)  # subclasses (that is, actual players) must implement this
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




NondeterministicAIPlayer = RandomPlayer
