# Quickstart

## Install

```bash
pip install steelflow
```

## Basic Usage

```python
from steelflow import DefaultRAGPipeline
from steelflow.rag import Retriever, Reranker

class MyRetriever(Retriever):
    def retrieve(self, query: str, **kwargs: object):
        return {"items": [{"id": "1", "text": "hello", "score": 1.0, "metadata": {}}], "latency_ms": 1.0}

class MyReranker(Reranker):
    def rerank(self, query: str, candidates):
        return {"items": candidates.items, "latency_ms": 1.0}

pipeline = DefaultRAGPipeline(retriever=MyRetriever(), reranker=MyReranker())
result = pipeline.run("What is SteelFlow?")
print(result)
```
