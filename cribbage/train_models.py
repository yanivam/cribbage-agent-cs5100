import numpy as np
import torch
from collections import deque

from cribbage.game import Deck
from cribbage.card import Card
from cribbage.score import score_hand, score_count
from random import choice, sample, randint


# Hyper-parameters for RL agent:
from QModel import LinearQNet, QTrainer
MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class DiscardGame():
    def __init__(self):
        self.player1_score = 0
        self.player2_score = 0
        self.deck = Deck()
        self.player1_hand = self.deck.draw(6)
        self.player2_hand = self.deck.draw(6)

def get_card_index(card):
    suit = card.suit
    value = card.rank
    return suit * 13 + value

def get_card_from_index(index):
    return Card(index)

def get_state(hand, discarded, previous_plays, dealer):
    '''
    Get the state of play as a one-hot encoded np.array.

    One input should represent whether the player is the dealer.
    52 inputs represent 1-hot vector of whether card is in hand.
    52 inputs represent "known" cards - i.e. the cards in the crib
    104 inputs represent the state of play - the first 13 represent the rank of the first card
    played, the second 13 represent the rank of the second, and so on.
    '''
    state = np.zeros(209)
    if dealer == 1:
        state[0] = 1

    # Set state of cards in hand:
    for card in hand:
        idx = get_card_index(card)
        idx = idx + 1  # Add one as we've already set the dealer state
        state[idx] = 1

    # Set state of "known cards" that were discarded to the crib:
    for card in discarded:
        idx = get_card_index(card)
        idx = idx + 52 + 1
        state[idx] = 1

    # Set the state of play:
    for i, card in enumerate(previous_plays):
        if i >= 8:
            pass
        card_idx = card.rank
        idx = card_idx + (i * 13) + 52 + 52 + 1
        state[idx] = 1

    return state

def discard(model, hand, dealer, num_games):
    discarded = []
    previous_plays = []

    # Sometimes act randomly for the first 5000 games. This will help increase exploration
    epsilon = 5000 - num_games
    if randint(0, 20000) < epsilon:
        card = hand[randint(0, len(hand)-1)]
        hand.remove(card)
        return card

    state = torch.Tensor(get_state(hand, discarded, previous_plays, dealer))
    output = model(state)
    vals, indices = output.sort(descending=True)
    for idx in indices:
        card = Card(idx)
        if card in hand:
            hand.remove(card)
            return card
    raise RuntimeError('Failed to discard a card')

def play(model, count, hand, dealer, discarded, previous_plays, num_games):
    # Sometimes act randomly for the first 5000 games. This will help increase exploration
    epsilon = 5000 - num_games
    if randint(0, 20000) < epsilon:
        card = hand[randint(0, len(hand)-1)]
        return card

    state = torch.Tensor(get_state(hand, discarded, previous_plays, dealer))
    output = model(state)
    vals, indices = output.sort(descending=True)
    for idx in indices:
        card = Card(idx)
        if card in hand and card.value + count <= 31:
            return card
    raise RuntimeError('Failed to play a card')


def remember(memory, state, action, reward, next_state, done):
    memory.append((state, action, reward, next_state, done))


def train_pegging_model(path_to_discard_model):
    '''
    Train a LinearQNet to play the "pegging" part of cribbage by making it play itself.
    Basically, deal each player 4 cards. Have them take turns playing, scoring the count along
    the way. The state is the state generated from the previous plays, the cards in hand, and the
    known cards. The reward is the points scored for each play. The next state is the resulting state
    from the initial state given the action, and the game ends when both players are out of cards.
    The idea is that if the cribbage player can play the pegging part of the game well they can
    play the total game well.
    '''
    discard_model = LinearQNet(209, 60, 52)
    discard_model.load_state_dict(torch.load(path_to_discard_model))
    discard_model.eval()
    model = LinearQNet(209, 60, 52)  # 2 Layer perceptron with 60 hidden nodes
    memory = deque(maxlen=MAX_MEMORY)
    trainer = QTrainer(model, lr=LR, gamma=0.9)
    train_iter = 1
    dealer_scores = 0
    pone_scores = 0


    while True:
        deck = Deck()
        player1_hand = list(deck.draw(6))  # represents Pone
        player2_hand = list(deck.draw(6))  # Represents Dealer
        player1_discards = []
        player2_discards = []
        scores = [0, 0]  # Keep track of total score

        # Perform discards:
        with torch.no_grad():
            for i in range(2):
                player1_discards.append(discard(discard_model, player1_hand, 0, 5001))
                player2_discards.append(discard(discard_model, player2_hand, 1, 5001))

        discards = [player1_discards, player2_discards]

        turn_map = {0: player1_hand, 1: player2_hand}  # map turn index to hand
        count = 0  # Keep track of pegging count
        turn = 0  # Index of whose turn it is - start with '0' for non-dealer
        plays = []  # Keep track of cards played so far
        plays_this_count = []  # Keep track of the cards played since last go
        go_has_been_said = False  # Keep track of whether GO has been said

        while player1_hand or player2_hand:
            player = turn_map[turn]
            playable_cards = [c for c in player if c.value + count <= 31]

            # Check to ensure there are playable cards
            if len(playable_cards) == 0:
                if go_has_been_said:
                    # If go has been said, reset the count and switch players
                    count = 0
                    turn = turn ^ 1
                    go_has_been_said = False
                    plays_this_count = []
                    continue
                go_has_been_said = True
                turn = turn ^ 1
                continue

            # Get the best playable card:
            state0 = get_state(player, discards[turn], plays, turn)
            card = play(model, count, player, turn, discards[turn], plays, train_iter)
            count += card.value  # Increment the count
            plays.append(card)  # Append the played card to the plays
            plays_this_count.append(card)
            player.remove(card)  # Remove the card from the hand
            reward = score_count(plays_this_count)  # Compute the score
            state1 = get_state(player, discards[turn], plays, train_iter)  # Get the resultant state

            # Check if play forces a go:
            opponent_cannot_play = len([c for c in turn_map[turn ^ 1] if c.value + count <= 31]) == 0
            if opponent_cannot_play:
                reward += 1  # Get an extra point for go

            # check if the player is out of cards:
            done = False
            if len(player) == 0:
                done = True

            # Keep track of total score:
            scores[turn] += reward

            # Remember the state-action-reward-state of this play
            remember(memory, state0, get_card_index(card), reward, state1, done)
            turn = turn ^ 1  # Switch turns
        pone_scores += scores[0]
        dealer_scores += scores[1]

        # Perform model training:
        if len(memory) > BATCH_SIZE:
            mini_sample = sample(memory, BATCH_SIZE)
        else:
            mini_sample = memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        trainer.train_step(states, actions, rewards, next_states, dones)
        train_iter += 1

        if train_iter % 100 == 0:
            print(f"Iteration {train_iter}")
            print(f"Average pegging score: {((dealer_scores + pone_scores) / 2) / train_iter}")

        if train_iter % 1000 == 0:
            torch.save(model.state_dict(), "./simple_pegging_model.pth")


def train_discard_model():
    model = LinearQNet(209, 60, 52)
    memory = deque(maxlen=MAX_MEMORY)
    trainer = QTrainer(model, lr=LR, gamma=0.9)
    train_iter = 1
    dealer_scores = 0
    pone_scores = 0
    hand_scores = 0
    crib_scores = 0

    while True:
        deck = Deck()
        player1_hand = list(deck.draw(6))
        player2_hand = list(deck.draw(6))
        turn_card = next(deck.draw(1))
        dealer = choice((0, 1))

        p1_state0 = get_state(player1_hand, [], [], dealer)
        p1_discard_1 = discard(model, player1_hand, dealer, train_iter)
        p1_state1 = get_state(player1_hand, [p1_discard_1], [], dealer)
        p1_discard_2 = discard(model, player1_hand, dealer, train_iter)
        p1_final_state = get_state(player1_hand, [p1_discard_1, p1_discard_2], [], dealer)

        p2_state0 = get_state(player2_hand, [], [], dealer ^ 1)
        p2_discard_1 = discard(model, player2_hand, dealer ^ 1, train_iter)
        p2_state1 = get_state(player2_hand, [p2_discard_1], [], dealer ^ 1)
        p2_discard_2 = discard(model, player2_hand, dealer ^ 1, train_iter)
        p2_final_state = get_state(player2_hand, [p2_discard_1, p2_discard_2], [], dealer ^ 1)

        crib = [p1_discard_1, p1_discard_2, p2_discard_1, p2_discard_2]
        crib_score = score_hand(crib, turn_card)
        p1_hand_score = score_hand(player1_hand, turn_card)
        p2_hand_score = score_hand(player2_hand, turn_card)
        hand_scores += (p1_hand_score + p2_hand_score)
        crib_scores += crib_score

        if dealer:
            p1_score = p1_hand_score
            p2_score = p2_hand_score
            dealer_scores += p1_score
            pone_scores += p2_hand_score
        else:
            p1_score = p1_hand_score
            p2_score = p2_hand_score
            dealer_scores += p2_score
            pone_scores += p1_hand_score

        remember(memory, p1_state0, get_card_index(p1_discard_1), p1_score, p1_state1, False)
        remember(memory, p1_state1, get_card_index(p1_discard_2), p1_score, p1_final_state, True)
        remember(memory, p2_state0, get_card_index(p2_discard_1), p2_score, p2_state1, False)
        remember(memory, p2_state1, get_card_index(p2_discard_2), p2_score, p2_final_state, True)

        if train_iter % 100 == 0:
            print(f"Iteration {train_iter} - Dealer was {['player1', 'player2'][dealer ^ 1]}")
            # print(f"p1 score: {p1_score}")
            # print(f"p2 score: {p2_score}")
            # print(f"Average dealer score: {dealer_scores / train_iter}")
            # print(f"Average pone score: {pone_scores / train_iter}")
            print(f"Average hand score: {hand_scores / (train_iter * 2)}")
            # print(f"Average crib score: {crib_scores / train_iter}")

        if train_iter % 1000 == 0:
            torch.save(model.state_dict(), "./simple_model.pth")

        # Perform model training:
        if len(memory) > BATCH_SIZE:
            mini_sample = sample(memory, BATCH_SIZE)
        else:
            mini_sample = memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        trainer.train_step(states, actions, rewards, next_states, dones)
        train_iter += 1

def test_discard_model_against_random_agent(path):
    model = LinearQNet(209, 60, 52)
    model.load_state_dict(torch.load(path))
    model.eval()

    with torch.no_grad():
        test_iter = 1
        model_scores = 0
        random_scores = 0
        model_scores_as_dealer = 0
        model_trials_as_dealer = 0
        model_scores_as_pone = 0
        model_trials_as_pone = 0

        while True:
            deck = Deck()
            model_hand = list(deck.draw(6))
            random_hand = list(deck.draw(6))
            turn_card = next(deck.draw(1))
            dealer = choice((0, 1))

            model_discard_1 = discard(model, model_hand, dealer, test_iter)
            model_discard_2 = discard(model, model_hand, dealer, test_iter)

            random_discard_1 = random_hand[randint(0, 5)]
            random_hand.remove(random_discard_1)
            random_discard_2 = random_hand[randint(0, 4)]
            random_hand.remove(random_discard_2)

            crib = [model_discard_1, model_discard_2, random_discard_1, random_discard_2]

            if dealer:
                dealer_score = score_hand(model_hand, turn_card) + score_hand(crib, turn_card)
                pone_score = score_hand(random_hand, turn_card)

                model_scores += dealer_score
                model_scores_as_dealer += dealer_score
                model_trials_as_dealer += 1

                random_scores += pone_score
            else:
                dealer_score = score_hand(random_hand, turn_card) + score_hand(crib, turn_card)
                pone_score = score_hand(model_hand, turn_card)

                model_scores += pone_score
                model_scores_as_pone += pone_score
                model_trials_as_pone += 1

                random_scores += dealer_score

            if test_iter % 1000 == 0:
                print(f"Num trials: {test_iter}")
                print(f"Average model score: {model_scores / test_iter}")
                print(f"Average random score: {random_scores / test_iter}")
                print(f"Average model score as dealer: {model_scores_as_dealer / model_trials_as_dealer}")
                print(f"Average model scores as pone: {model_scores_as_pone / model_trials_as_pone}")

            test_iter += 1


if __name__ == '__main__':
    # train_discard_model()
    # test_discard_model_against_random_agent("./simple_model.pth")
    train_pegging_model("./simple_model.pth")
