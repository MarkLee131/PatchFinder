# PatchTracer
PatchTracer: A Two-Phase Approach to Security Patch Tracing for Disclosed Vulnerabilities in Open Source Software

## Overview
PatchTracer operates in two distinct phases:

1. Efficient **initial retrieval** via TF-IDF.
2. Powerful **re-ranking** via fine-tuning CodeReviewer.

## Abstract
Open source software (OSS) vulnerabilities are increasingly prevalent, emphasizing the importance of security patches. However, in widely used security platforms like NVD, a substantial number of CVE records still lack trace links to patches. Although rank-based approaches have been proposed for security patch tracing, they heavily rely on handcrafted features in a single-step framework, which limits their effectiveness.

In this paper, we propose PatchTracer, a two-phase framework with end-to-end correlation learning for better-tracing security patches. In the **initial retrieval** phase, we employ TF-IDF, an efficient and powerful information retrieval method, to narrow down the search space by extracting those commits as candidates that are lexically similar to the CVE descriptions. Afterward, in the **re-ranking** phase, we design an end-to-end architecture under the supervised fine-tuning paradigm for learning the semantic correlations between CVE descriptions and commits. In this way, we can automatically rank the candidates based on their correlation scores while maintaining low computation overhead. We evaluated our system against 4,789 CVEs from 532 OSS projects. The results are highly promising: PatchTracer achieves a Recall@10 of 77.92% and a Mean Reciprocal Rank (MRR) of 0.7727. Moreover, the manual effort@10 required is curtailed to 2.99, marking a 1.88 times improvement over current leading methods. When applying PatchTracer in practice, we initially identified **172** patch commits (average rank at **2.89**) and submitted them to the official, **135** of which have been confirmed by CVE Numbering Authorities.


