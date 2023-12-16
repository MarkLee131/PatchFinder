# PatchFinder
PatchFinder A Two-Phase Approach to Security Patch Tracing for Disclosed Vulnerabilities in Open Source Software

## Overview
PatchFinder operates in two distinct phases:

1. Efficient **initial retrieval** via Hybrid Retriever.
2. Powerful **re-ranking** via fine-tuning CodeReviewer.

## Abstract
Open-source software (OSS) vulnerabilities are increasingly prevalent, emphasizing the importance of security patches. However, in widely used security platforms like NVD, a substantial number of CVE records still lack trace links to patches. Although rank-based approaches have been proposed for security patch tracing, they heavily rely on handcrafted features in a single-step framework,
which limits their effectiveness.

In this paper, we propose PatchFinder, a two-phase framework with end-to-end correlation learning for better-tracing security patches. In the initial retrieval phase, we employ a hybrid patch retriever to account for both lexical and semantic matching based on the code changes and the description of a CVE, to narrow down the search space by extracting those commits as candidates that are similar to the CVE descriptions. Afterwards, in the re-ranking phase, we design an end-to-end architecture under the supervised fine-tuning paradigm for learning the semantic correlations be-
tween CVE descriptions and commits. In this way, we can automatically rank the candidates based on their correlation scores while maintaining low computation overhead. We evaluated our system
against 4,789 CVEs from 532 OSS projects. The results are highly promising: PatchFinder achieves a Recall@10 of 80.63% and a Mean Reciprocal Rank (MRR) of 0.7951. Moreover, the manual effort@10 required is curtailed to 2.77, marking a 1.94 times improvement over current leading methods. When applying PatchFinder in practice, we initially identified 172 patch commits (average rank at 1.65) and submitted them to the official, 135 of which have been confirmed by CVE Numbering Authorities.

![overview of out approach](./overview-github.png)
