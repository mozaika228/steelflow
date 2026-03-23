# RAG Recipes

## Hybrid Retrieval

```python
from steelflow.rag import HybridMerger, HybridWeights, RetrievalItem

merger = HybridMerger(HybridWeights(dense=0.6, sparse=0.4))

dense = [RetrievalItem(id="a", text="dense", score=0.8, metadata={})]
sparse = [RetrievalItem(id="a", text="sparse", score=0.5, metadata={})]

merged = merger.merge(dense, sparse, top_k=10)
print(merged)
```
