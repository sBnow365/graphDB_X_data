# Optional GNN Extension (not part of the baseline)

The baseline is **not** GNN-based. Louvain and PageRank are classical graph algorithms; Graph RAG
uses graph query results as evidence. None of them learns neural representations.

## What a GNN is
A neural network that computes a vector for each node by **message passing**: each node aggregates its
neighbours' vectors, combines them with its own, and repeats for k layers (k-hop information).
* **GCN** (Kipf & Welling, ICLR 2017): normalised neighbourhood averaging.
* **GraphSAGE** (Hamilton et al., NeurIPS 2017): samples neighbours, inductive (works on new nodes).
* **GAT** (Veličković et al., ICLR 2018): attention weights over neighbours.

## How it could complement this project
Node features = mean text embedding of a user's posts (+ activity counts); edges = the interaction
projection. Possible objectives: user-level "posts hate-labelled content" prediction (labels from the
classifier: beware circularity), bot detection on TwiBot-22 (its intended task), or link prediction.
Learned embeddings could also enrich retrieval (similar-user search).

## Evaluation against the baseline
Same splits (users disjoint), compare a text-only model (mean embedding + logistic regression) vs
GNN; report macro-F1 and variance over seeds. Isolate code in `experiments/gnn/` (PyTorch Geometric
or DGL). Neo4j GDS also provides FastRP/GraphSAGE embeddings **[verify edition availability]**.
