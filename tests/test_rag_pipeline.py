from steelflow.rag import DefaultRAGPipeline, Retriever


class WeirdRetriever(Retriever):
    def retrieve(self, query: str, **kwargs: object):
        return {
            "items": [
                {"id": 1, "text": 123, "score": "bad", "metadata": ["x"]},
            ],
            "latency_ms": 1.0,
        }


def test_pipeline_normalizes_items() -> None:
    pipeline = DefaultRAGPipeline(retriever=WeirdRetriever())
    out = pipeline.run("q")

    item = out["retrieved"][0]
    assert item.id == "1"
    assert item.text == "123"
    assert item.score == 0.0
    assert item.metadata == {}
