import numpy as np
import sys
import json
import os

from Gambler import Gambler

# Check command-line arguments
if len(sys.argv) != 3:
    print(f"Usage: python3 {sys.argv[0]} <dice_file> <rolls>")
    sys.exit(1)

# The path to the json describing the dice and croupier
game_path = sys.argv[1]

# How many rolls in this session?
num_rolls = int(sys.argv[2])

# Read in the dice json
with open(game_path, "r") as f:
    game_dict = json.load(f)

# Convert lists to numpy arrays
payout = np.array(game_dict["payout"], dtype=np.float64)
transitions = np.array(game_dict["transitions"], dtype=np.float64)
initial = np.array(game_dict["initial"], dtype=np.float64)
dice = np.array(game_dict["dice"], dtype=np.float64)

(dice_count, dice_sides) = dice.shape

# Create an instance of the Gambler class
gambler = Gambler(dice, payout, transitions, initial)

# Show some basic statistics
stationary_distribution = gambler.stationary_distribution()
print(f"Here is the stationary distribution of die use: \n{stationary_distribution}\n")

expected_winnings_by_die = gambler.expected_winnings_by_die()
print(f"Expected winnings per roll for each die:\n{expected_winnings_by_die}\n")

expected_winnings_per_roll_bet_every_roll = (
    gambler.expected_winnings_per_roll_bet_every_roll()
)
print(
    f"Expected per roll winnings betting every roll: ${expected_winnings_per_roll_bet_every_roll:,.2f}"
)

expected_winnings_per_roll_bet_on_good_dice = (
    gambler.expected_winnings_per_roll_bet_on_good_dice()
)
print(
    f"Expected per roll winnings if you knew which die was being used: ${expected_winnings_per_roll_bet_on_good_dice:,.2f}"
)

print(
    f"Expected total winnings if you bet on every roll: ${expected_winnings_per_roll_bet_every_roll * num_rolls:,.2f}"
)
print(
    f"Expected total winnings if you knew which die was being used: ${expected_winnings_per_roll_bet_on_good_dice * num_rolls:,.2f}"
)

# For tracking optimal play
total_could_have_won = 0.0

# For tracking actual play
total_did_win = 0.0

# For tracking optimistic playing
total_dumb_win = 0.0

# Choose a die using initial distribution
current_die = np.random.choice(dice_count, 1, p=initial)[0]

for i in range(num_rolls):

    # Note whether this was a good die to bet on
    should_bet = expected_winnings_by_die[current_die] > 0.0

    # Ask the gambler if it wants to bet
    is_betting = gambler.will_bet()

    # Roll the die
    die_p = dice[current_die]
    current_side = np.random.choice(dice_sides, 1, p=die_p)[0]

    # Tell the gambler what came up
    gambler.update(current_side)

    # How much would the gambler have won on this roll?
    # (Might be negative!)
    current_payout = payout[current_side]

    # Update results for the gambler
    if is_betting:
        total_did_win += current_payout

    # Update results for all-knowing cheat
    if should_bet:
        total_could_have_won += current_payout

    # Update results for a dumb optimistic player
    total_dumb_win += current_payout

    # Deal with the possible transition to a new state
    current_t = transitions[current_die, :]
    current_die = np.random.choice(dice_count, 1, p=current_t)[0]

# Print the final results
print(f"Total would have won if bet on every roll: ${total_dumb_win:,.2f}")
print(
    f"Total would have won if you knew the die was a winner: ${total_could_have_won:,.2f}"
)
print(f"Did win: ${total_did_win:,.2f}")
per_roll_winnings = total_did_win / num_rolls
print(f"Average per roll winnings: ${per_roll_winnings}")
