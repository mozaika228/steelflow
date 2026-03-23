from steelflow.rag import DefaultRAGPipeline, Retriever


class ToyRetriever(Retriever):
    def retrieve(self, query: str, **kwargs: object):
        return {
            "items": [
                {"id": "1", "text": "SteelFlow is local-first", "score": 1.0, "metadata": {}}
            ],
            "latency_ms": 1.2,
        }


pipeline = DefaultRAGPipeline(retriever=ToyRetriever())
print(pipeline.run("What is SteelFlow?"))
