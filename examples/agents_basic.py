from steelflow.agents import FunctionStep, SimplePipeline
from steelflow.contracts import AgentStepResult


def planner(task: str, tools, state):
    return AgentStepResult(name="planner", output=f"plan for {task}", latency_ms=1.0, metadata={})


def writer(task: str, tools, state):
    return AgentStepResult(name="writer", output=f"draft for {task}", latency_ms=2.0, metadata={})


pipeline = SimplePipeline(steps=[
    FunctionStep("planner", planner),
    FunctionStep("writer", writer),
])

print(pipeline.run("marketing brief", tools=[]))
