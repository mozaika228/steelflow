from steelflow.rag import HybridMerger, HybridWeights, RetrievalItem


def test_hybrid_merge_weighting() -> None:
    dense = [RetrievalItem(id="a", text="A", score=1.0, metadata={})]
    sparse = [RetrievalItem(id="a", text="A", score=0.5, metadata={})]

    merger = HybridMerger(HybridWeights(dense=0.7, sparse=0.3))
    merged = merger.merge(dense, sparse, top_k=1)

    assert len(merged) == 1
    assert merged[0].id == "a"
    assert abs(merged[0].score - (1.0 * 0.7 + 0.5 * 0.3)) < 1e-6
