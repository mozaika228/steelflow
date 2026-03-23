# Agents

Lean agent pipelines can be defined as a list of steps. Each step is a small unit of work with explicit inputs.

```python
from steelflow.agents import FunctionStep, SimplePipeline
from steelflow.contracts import AgentStepResult


def planner(task: str, tools, state):
    return AgentStepResult(name="planner", output=f"plan for {task}", latency_ms=1.0, metadata={})


def executor(task: str, tools, state):
    return AgentStepResult(name="executor", output=f"executed {task}", latency_ms=2.0, metadata={})

pipeline = SimplePipeline(steps=[
    FunctionStep("planner", planner),
    FunctionStep("executor", executor),
])

results = pipeline.run("build report", tools=[])
print(results)
```
