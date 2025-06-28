from enum import Enum
import pyspiel

NUM_PLAYERS = 4
HAND_SIZE = 18
CHIEN_SIZE = 6
CHIEN_ID = NUM_PLAYERS + 1
NUM_TRICKS = 18
DECK_SIZE = 78


class Phase(Enum):
    BIDDING = 1
    DECLARE = 2
    CHIEN = 3
    CHELEM = 4
    POIGNEE = 5
    TRICK = 6
    END = 7


POINTS_PER_BOUT = {
    3: 36,
    2: 41,
    1: 46,
    0: 51
}
ALPHA = {
    3: 0.25,
    2: 0.25,
    1: 0.25,
    0: 0.25
}


MIN_POIGNEE = 10
POIGNEE_BONUS = {
    10: 20,
    13: 30,
    15: 40,
}
BETA = {
    0: 0.25,
    10: 0.25,
    13: 0.25,
    15: 0.25,
}


FOU = 500
PETIT = 501
MONDE = 521
BOUTS = [FOU, PETIT, MONDE]
TRUMPS = 22

TRUMP = 5
HEART = 4
DIAMOND = 3
CLUB = 2
SPADE = 1

JACK = 11
KNIGHT = 12
QUEEN = 13
KING = 14


MAX_POINTS = (3 * 4.5) + (4 * 4.5) + (4 * 3.5) + (4 * 2.5) + (3 * 1.5)
MIN_POINTS = (DECK_SIZE // NUM_PLAYERS) * 0.5

NUM_BOUTS = len(BOUTS)
NUM_TRUMPS = TRUMPS

PASS = 600
PETIT = 601
GARDE = 602
GARDE_SANS = 603
GARDE_CONTRE = 604
GAMMA = {
    PASS: 0.2,
    PETIT: 0.2,
    GARDE: 0.2,
    GARDE_SANS: 0.2,
    GARDE_CONTRE: 0.2
}
BIDS = [PASS, PETIT, GARDE, GARDE_SANS, GARDE_CONTRE]
NUM_BIDS = len(BIDS)
NUM_BID_SIZE = NUM_PLAYERS

DECLARE_NONE = 700
DECLARE_CHELEM = 701
DECLARE_POIGNEE = 702
DECLARES = [DECLARE_NONE, DECLARE_CHELEM, DECLARE_POIGNEE]
NUM_DECLARES = len(DECLARES)
NUM_DECLARES_SIZE = 2 * NUM_PLAYERS


GAME_TYPE = pyspiel.GameType(
    short_name="french_tarot",
    long_name="French Tarot",
    dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
    chance_mode=pyspiel.GameType.ChanceMode.EXPLICIT_STOCHASTIC,
    information=pyspiel.GameType.Information.IMPERFECT_INFORMATION,
    utility=pyspiel.GameType.Utility.ZERO_SUM,
    reward_model=pyspiel.GameType.RewardModel.TERMINAL,
    max_num_players=NUM_PLAYERS,
    min_num_players=NUM_PLAYERS,
    provides_information_state_string=False,
    provides_information_state_tensor=False,
    provides_observation_string=True,
    provides_observation_tensor=True,
    parameter_specification={}
)

GAME_INFO = pyspiel.GameInfo(
    num_distinct_actions=DECK_SIZE,
    max_chance_outcomes=NUM_BIDS + NUM_DECLARES,
    num_players=NUM_PLAYERS,
    min_utility=-1.0,
    max_utility=1.0,
    utility_sum=0.0,
    max_game_length=(
        NUM_PLAYERS * NUM_TRICKS +
        CHIEN_SIZE +
        NUM_BID_SIZE +
        NUM_DECLARES_SIZE)
)


MASK_DECK_SIZE = DECK_SIZE
MASK_NUM_TRICK_SIZE = NUM_PLAYERS + 1
MASK_NUM_TRICKS_SIZE = MASK_NUM_TRICK_SIZE * NUM_TRICKS
MASK_CURRENT_PLAYER_SIZE = 1
MASK_TAKER_PLAYER_SIZE = 1
MASK_BID_SIZE = 4
MASK_DECLARATIONS_SIZE = 4
MASK_PHASE_SIZE = 1

MASK_SIZE = [MASK_DECK_SIZE,
             MASK_NUM_TRICK_SIZE,
             MASK_NUM_TRICKS_SIZE,
             MASK_CURRENT_PLAYER_SIZE, MASK_TAKER_PLAYER_SIZE, MASK_BID_SIZE,
             MASK_DECLARATIONS_SIZE, MASK_PHASE_SIZE]

MASK = {
    'known_cards': 0,
    'current_trick': 1,
    'known_tricks': 2,
    'current_player': 3,
    'taker_player': 4,
    'bid': 5,
    'declarations': 6,
    'phase': 7,
}

MASK_SIZE_TOTAL = sum(MASK_SIZE)
