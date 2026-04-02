# Enron Email Friend Recommendation System
 
A Graph Neural Network built on the Enron email dataset that recommends people you likely know but haven't directly emailed, and identifies tight-knit groups within the network.
 
---
 
## What's in here
 
- **Data preprocessing** — parsing raw email headers, cleaning addresses, exploding multi-recipient rows
- **Graph construction** — building a bidirectional user-user interaction graph from 104k edges across 36k users
- **Train/val/test split** — 80/10/10 random edge split
- **FriendGNN model** — LightGCN-based graph neural network with learned 64-dim user embeddings
- **BPR loss** — pairwise ranking loss for implicit feedback (no ratings, just interactions)
- **Friend recommendations** — dot product similarity on learned embeddings, existing contacts masked out
- **Clique analysis** — NetworkX-based maximal clique detection with Jaccard scoring for expansion candidates
 
---
 
## Dataset
Enron email corpus — 517,401 emails, available on Kaggle as `emails.csv`.
 
## Stack
`PyTorch` · `PyTorch Geometric` · `NetworkX` · `pandas` · `NumPy`
