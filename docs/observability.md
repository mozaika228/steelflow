# Observability

Observability is required by design. Use the facade to wire real exporters later.

```python
from steelflow.observability import Observability

obs = Observability.default()
span = obs.tracer.start_span("demo")
obs.metrics.observe("tokens_per_sec", 42.0)
obs.logger.info("demo complete", ok=True)
span.end()
```
