"""Produces the verdict-accuracy numbers in RESULTS.md, run against the
offline demo evidence corpus (see evidence/*.txt).

    python scripts/measure_agent.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agent import check_claim

GOLD = [
    ("The Eiffel Tower is located in Paris, France.", "Supported"),
    ("The Eiffel Tower is located in London, England.", "Contradicted"),
    ("The Great Wall of China is visible from space with the naked eye.", "Contradicted"),
    ("Napoleon Bonaparte was extremely short compared to other men of his time.", "Contradicted"),
    ("Water boils at 100 degrees Celsius at sea level.", "Supported"),
    ("The Moon is Earth's only natural satellite.", "Supported"),
    ("Mount Everest is the highest mountain on Earth above sea level.", "Supported"),
    ("Bananas are the most popular fruit on Mars.", "Unverifiable"),
    ("The stock market rose sharply yesterday.", "Unverifiable"),
    ("A new species of beetle was discovered in Peru last week.", "Unverifiable"),
]


def main():
    correct = 0
    start = time.perf_counter()
    for claim, expected in GOLD:
        v = check_claim(claim)
        ok = v.verdict == expected
        correct += ok
        mark = "OK  " if ok else "MISS"
        print(f"{mark} expected={expected:<12} got={v.verdict:<12} ({v.confidence})  {claim}")
    elapsed = time.perf_counter() - start

    n = len(GOLD)
    print(f"\naccuracy: {correct}/{n} = {correct/n:.2f}")
    print(f"{n} claims in {elapsed:.2f}s ({elapsed/n*1000:.0f}ms/claim)")


if __name__ == "__main__":
    main()
