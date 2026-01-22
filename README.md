# GBG
Federated learning (FL) collaboratively trains a
global model across multiple clients without sharing local data,
effectively utilizing data while preserving privacy. However,
real-world data are often non-independently and identically
distributed (non-IID) and heterogeneous across clients, making
standard FL less suitable for practical deployment. To tackle
these challenges, we propose a novel framework named GBG.
First, it introduces a novel grouping mechanism that clusters
heterogeneous clients into different groups. Clients in these
groups perform different training tasks. Second, it employs Sym-
metric Balanced Incomplete Block Design (SBIBD) to construct
intra-group blocks and establishes a new training paradigm
called block training. Finally, by incorporating mutual learning,
the GBG framework enables the effective development of both
personalized block-level models and a global model. Moreover,
we demonstrate the convergence of block training in combination
with existing works. Additional experiments further demonstrate
that the GBG framework achieves favorable results in both
testing accuracy and training error
