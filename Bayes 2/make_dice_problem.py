import sys
import random

def rollsum(num_dice, num_sides):
    return sum(random.randint(1, num_sides) for _ in range(num_dice))

if len(sys.argv) != 6:
    print(f"Usage: python3 {sys.argv[0]} <num_dice> <num_sides> <rounds> <minlen> <maxlen>")
    sys.exit(1)

num_dice = int(sys.argv[1])
num_sides = int(sys.argv[2])
rounds = int(sys.argv[3])
minlen = int(sys.argv[4])
maxlen = int(sys.argv[5])

if num_dice <= 1 or num_sides <= 1 or rounds <= 1 or minlen <= 1 or maxlen <= 1:
    print("All args must be positive integers")
    sys.exit(1)

prob_path = f"prob-{num_dice}-{maxlen}.txt"

with open(prob_path, 'w') as f:
    print(f"{num_dice} {num_sides} {rounds}", file=f)

    for i in range(rounds):
        rollcount = minlen + (i * (maxlen - minlen)) // rounds

        original_sum = random.randint(num_dice, num_dice * num_sides)
        print(f"Original:{original_sum}")

        for j in range(rollcount):
            x = rollsum(num_dice, num_sides)
            if x > original_sum:
                next = "H"
            elif x < original_sum:
                next = "L"
            else:
                next = "E"
            print(next,file=f,end="")
        print("", file=f)
