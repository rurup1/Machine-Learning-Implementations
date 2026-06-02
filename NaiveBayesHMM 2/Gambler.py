import jax.numpy as jnp


class Gambler:

    def __init__(
        self,
        dice: jnp.ndarray,
        payout: jnp.ndarray,
        transitions: jnp.ndarray,
        initial: jnp.ndarray,
    ):
        # Note the arguments that were passed in
        self.dice = dice
        self.payout = payout
        self.transitions = transitions
        self.initial = initial

        # Cache this -- you will need it a lot
        self.expected_winnings_for_each_die = ## Your code here

        # Your belief about what die is being used on the next roll
        # This will be updated after every roll
        self.state_p = initial.copy()

    # Current beliefs about the probability that each die is about to be rolled
    # Used by the autograder
    def current_beliefs(self) -> jnp.ndarray:
        return self.state_p

    # Assume Markov Chain is ergodic, return the stationary_distribution
    def stationary_distribution(self) -> jnp.ndarray:
        ## Your code here

    # Average winnings for each roll of each die
    def expected_winnings_by_die(self) -> jnp.ndarray:
        return self.expected_winnings_for_each_die

    # If you bet on every roll, what would you expect your average
    # winnings per roll to be?
    def expected_winnings_per_roll_bet_every_roll(self) -> jnp.float64:
    ## Your code here

    # If the croupier told you what die he was using, what would you expect your average
    # winnings per roll to be? (Include in "per roll" all the rolls, not just the ones you would bet on.)
    def expected_winnings_per_roll_bet_on_good_dice(self) -> jnp.float64:
    ## Your code here

    def will_bet(self) -> bool:
        # Assuming I am right about the state_p, what should I expect to win?
        expected_winning = ## Your code here

        # If that is positive, I will bet
        return expected_winning > 0.0

    # The croupier just rolled a 'side'
    # Update your beliefs
    def update(self, side: int):
    ## Your code here
