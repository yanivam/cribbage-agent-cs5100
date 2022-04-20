import pytest 

from .players import WinGame, Player, GreedyAgentPlayer
from .card import hand_from_str, card_from_str


def test_peg():
    player = Player()
    player.peg(13)
    assert 13 == player.score


def test_win():
    player = Player()
    with pytest.raises(WinGame):
        player.peg(150)


@pytest.mark.slow
def test_greedy_agent_chooses_bad_crib():
    player = GreedyAgentPlayer()
    player.hand, turn_card = hand_from_str("5d 5h Ad 3s 8h")
    discards = player.ask_for_discards()
    assert all(x.value != 5 for x in discards)