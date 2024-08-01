# PatchFinder [![arXiv](https://img.shields.io/badge/arXiv-<2407.17065>-<COLOR>.svg)](https://arxiv.org/abs/2407.17065) 
PatchFinder: A Two-Phase Approach to Security Patch Tracing for Disclosed Vulnerabilities in Open Source Software (Published @ [ISSTA 2024](https://2024.issta.org/details/issta-2024-papers/48/PatchFinder-A-Two-Phase-Approach-to-Security-Patch-Tracing-for-Disclosed-Vulnerabili))


## Overview
PatchFinder operates in two distinct phases:

1. Efficient **initial retrieval** via Hybrid Retriever.
2. Powerful **re-ranking** via fine-tuning CodeReviewer.

## Abstract
Open source software (OSS) vulnerabilities are increasingly prevalent, emphasizing the importance of security patches. However, in widely used security platforms like NVD, a substantial number of CVE records still lack trace links to patches. Although rank-based approaches have been proposed for security patch tracing, they heavily rely on handcrafted features in a single-step framework, which limits their effectiveness.

In this paper, we propose PatchFinder, a two-phase framework with end-to-end correlation learning for better-tracing security patches. In the initial retrieval phase, we employ a patch retriever to account for both lexical and semantic matching based on the raw source code, an efficient and powerful information retrieval method, to narrow down the search space by extracting those commits as candidates that are similar to the CVE descriptions. Afterward, in the re-ranking phase, we design an end-to-end architecture under the supervised fine-tuning paradigm for learning the semantic correlations between CVE descriptions and commits. In this way, we can automatically rank the candidates based on their correlation scores while maintaining low computation overhead. We evaluated our system against 4,789 CVEs from 532 OSS projects. The results are highly promising: PatchFinder achieves a Recall@10 of 80.63% and a Mean Reciprocal Rank (MRR) of 0.7951. Moreover, the manual effort@10 required is curtailed to 2.77, marking a 2.03 times improvement over current leading methods. 
When applying PatchFinder in practice, we initially identified 533 patch commits (average rank at 1.65) and submitted them to the official, 482 of which have been confirmed by CVE Numbering Authorities.


![overview of out approach](./overview-github.png)


## Cite us

### BibTeX

```
@inproceedings{li2024patchfinder,
  title={PatchFinder: A Two-Phase Approach to Security Patch Tracing for Disclosed Vulnerabilities in Open Source Software},
  author={Li, Kaixuan and Zhang, Jian and Chen, Sen and Liu, Han and Liu, Yang and Chen, Yixiang},
  booktitle={Proceedings of the 33rd ACM SIGSOFT International Symposium on Software Testing and Analysis},
  year={2024}
}
```
