# PatchSleuth
PatchSleuth: Sleuthing for Disclosed Security Patch Commits in Open-Source Software

## Overview
PatchSleuth operates in two distinct phases:

1. Scope Reduction: Utilizing TF-IDF or BM25 algorithms, PatchSleuth identifies the most pertinent commits, effectively narrowing down the scope.

2. Commit Ranking: A neural network is employed to rank these commits, assessing them based on their probability of being a security patch commit.