from copy import deepcopy
from .score import peg_val
from .card import Deck
import numpy as np
#build a minimax tree and returns the best move
#tree with designated depth, currently 3 (predicitng one move from opponent)
class minimaxTree:


    class node:
        def __init__(self, card, utility):
            self.children = []
            self.utility = utility
            self.myScore = 0
            self.opponantScore = 0
            self.sumFromPlay = 0
            self.card = card

        def addChild(self, childNode):
            self.children.append(childNode)

        def debug(self):
            print("Debugging Info for Normal Node:")
            print("Children: ", self.children)
            print("Utility: ", self.utility)
            print("Card:", self.card)

        def getChildren(self):
            return self.children

        def getCard(self):
            return self.card

        def getSumFromPlay(self):
            return self.sumFromPlay

        def setSumFromPlay(self, sum):
            self.sumFromPlay = sum

        def getUtility(self):
            return self.utility

    class chanceNode:
        def __init__(self, card, probability):
            self.children = []
            self.utility = 0
            self.card = card
            self.probability = probability

        def debug(self):
            print("Debugging Info for Chance Node:")
            print("Children: ", self.children)
            print("Utility: ", self.utility)
            print("Card:", self.card)

        def addChild(self, childNode):
            self.children.append(childNode)

        def getChildren(self):
            return self.children

        def getCard(self):
            return self.card

        def getUtility(self):
            return self.utility

    def scorePegging(self, hand, sequence, current_sum, card):
        # Calculate the score for playing any specific card
        if len(sequence) > 3:
            cards = [peg_val(sequence[len(sequence) - 1]), peg_val(sequence[len(sequence) - 2]), peg_val(sequence[len(sequence) - 3])]
            cards.sort()
            if cards[0] + 1 in cards and cards[1] + 1 in cards:
                if peg_val(card) == (cards[len(cards) - 1]) + 1 and peg_val(card) + current_sum <= 31:
                    return 4

        # Check 3rd card sequence
        if len(sequence) > 2:
            cards = [peg_val(sequence[len(sequence) - 1]), peg_val(sequence[len(sequence) - 2])]
            cards.sort()
            if cards[0] + 1 in cards:
                if peg_val(card) == cards[len(cards) - 1] + 1 and peg_val(card) + current_sum <= 31:
                    return 3

        # Get sum to 15
            if peg_val(card) + current_sum == 15:
                return 2

        # Play same rank
        if len(sequence) > 0:
            check = sequence[len(sequence) - 1]
            if card == check and peg_val(card) + current_sum <= 31:
                return 2
        return 0

    # Initializes an expectimax tree of a specified depth
    def __init__(self, hand, sequence, current_sum, tree_depth):
        self.root = self.node(None, 0)

        # Consider all cards in hand that are legal moves
        legal_moves = []
        for card in hand:
            # print(card)
            if card.value + current_sum <= 31:
                legal_moves.append(card)
                newNode = self.node(card, self.scorePegging(hand, sequence, current_sum, card))
                newNode.sumFromPlay = current_sum + card.value
                self.root.addChild(newNode)

        # print("legal moves: ", legal_moves)
        # print("Root's children: ", root.children)


        # Get deck of cards
        whole_deck = []
        deck = Deck()
        for i in deck.cards:
            whole_deck.append(i)

        # Remove known cards from deck
        for i in hand:
            whole_deck.remove(i)
        for i in sequence:
            whole_deck.remove(i)

        # For each level 2 node, put in prob nodes with legal moves
        for cnode in self.root.children:
            # Get rid of illegal or used cards
            card_deck = deepcopy(whole_deck)
            if cnode.getCard() in card_deck:
                card_deck.remove(cnode.getCard())
            for card in card_deck:
                if cnode.getSumFromPlay() + card.value > 31 and card in card_deck:
                    card_deck.remove(card)

            # Make new nodes in children of current node using prob nodes
            for card in card_deck:
                newChanceNode = self.chanceNode(card, 1/len(card_deck))
                cnode.addChild(newChanceNode)

        # Go through all prob nodes and add regular nodes for opponent
        for n in self.root.getChildren(): #cnode, dealer node
            expected_utility = []
            for m in n.getChildren(): #chanceNodes
                testHand = deepcopy(hand)
                testHand.remove(n.getCard())
                testSequence = deepcopy(sequence)
                testSequence.append(n.getCard())
                utility = -1*(self.scorePegging(testHand, testSequence, current_sum + n.getSumFromPlay(), m.getCard()))
                newNode = self.node(m, utility)
                m.addChild(newNode)
                expected_utility.append(utility) 
            n.utility = np.mean(expected_utility)

        



        

    # Searches the expectimax tree and recommends the card to be played
    # 0 for  no risk, 1 for medium risk, 2 for high risk
    def recommendCard(self, risk):
        if len(self.root.getChildren()) == 0:
            return None
        #print(self.root.getChildren())
        lowestNode = self.root.getChildren()[0]

        lowestNodeScore = float('-inf')
        for cnode in self.root.getChildren():
            # Opponant will play whatever card is most negative utility for agent
            # Find min node
            # currMin = 1000
            # for dnode in cnode.getChildren():
            #     if dnode.getChildren()[0].getUtility() < currMin:
            #         currMin = dnode.getChildren()[0].getUtility()
            if (cnode.getUtility() ) > lowestNodeScore: #+ currMin
                lowestNode = cnode
                lowestNodeScore = cnode.getUtility()

        return lowestNode.getCard()



    