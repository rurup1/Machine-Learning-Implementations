from Belief import Belief

x = Belief(2, 6)

print(x.show_array(x.priors, 'P(d) | '))

for key, value in x.letter_likelihoods.items():
    print(x.show_array(value, f"P({key}|d)"))

