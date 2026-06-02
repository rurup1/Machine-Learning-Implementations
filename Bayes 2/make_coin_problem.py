from scipy.stats import beta
import random
import sys

if len(sys.argv) != 3:
    print(f"Usage: python3 {sys.argv[0]} <expnum> <outfilename>")
    exit(-1)

ALPHA = 5
BETA = 3

credibles = [x * 0.05 for x in range(10, 19)]
cred = random.choice(credibles)

n = int(sys.argv[1])
outpath = sys.argv[2]
theta = beta.rvs(ALPHA, BETA)

print(f"Creating {n} experiments: \n\ttheta = {theta:.4f}\n\tcred={cred:.2f}")
with open(outpath, "w") as outfile:

    print(f"{cred:.2f}", file=outfile)

    for _ in range(n):
        r = random.randrange(4, 15)
        successes = 0
        attempts = 0
        while successes < r:
            attempts += 1
            if random.random() < theta:
                successes += 1
        print(f"{r} {attempts}", file=outfile)

print(f"Created {outpath}")
