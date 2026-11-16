import asyncio
import sys
from pathlib import Path
import logfire
from aiolimiter import AsyncLimiter
from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelMessage, RunUsage, UsageLimits
from dotenv import load_dotenv
from pydantic_ai.mcp import MCPToolset
from fastmcp.client.transports import StdioTransport
from llmbot.rate_limit import RateLimitedModel

# Load API keys from the repository's .env, wherever the agent is started from
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

# Defining the Stdio connection to Webots MCP server script
# This script runs the FastMCP server, exposing the robot's capabilities as tools.
transport = StdioTransport(
    command=sys.executable,
    args=[str(Path(__file__).resolve().parent / "panda_mcp_server.py")]
)

logfire.configure(send_to_logfire='if-token-present')
logfire.instrument_pydantic_ai()


executor_model='openai-chat:Qwen/Qwen3.8-27B-NVFP4'
planner_model='google:gemini-3.5-flash-lite'

# Wrap the transport in an MCPToolset
toolset = MCPToolset(transport)

rate_limiter = AsyncLimiter(max_rate=1, time_period=5)
PLANNER_LIMITS = UsageLimits(request_limit=15)
EXECUTOR_LIMITS = UsageLimits(request_limit=30)

# Creating filtered toolset for the agents

PLANNER_TOOLS = {"get_area_position","get_camera_image","get_gripper_state","get_objects","get_target_position"}
EXECUTOR_TOOLS = {"pick","place","go_home"}
planner_toolset = toolset.filtered(lambda ctx, tool: tool.name in PLANNER_TOOLS)
executor_toolset = toolset.filtered(lambda ctx, tool: tool.name in EXECUTOR_TOOLS)

# Define output type for planner agent
class Step(BaseModel):
    """One tool call of the plan."""
    step: int = Field(..., description=" The number of the current step")
    tool: str = Field(..., description="The tool being used for the current step")
    object_id: str | None = Field(default=None,description="if the called tool requires an object id, write it here")
    target_area: str | None = Field(default=None,description="If the called tool needs a destination area specify it here")
    x: float | None = Field(default=None,description="If the called tool requires a position, specify its x coordinate here")              
    y: float | None = Field(default=None,description="If the called tool requires a position, specify its y coordinate here")
    z: float | None = Field(default=None,description="If the called tool requires a position, specify its z coordinate here")
    yaw: float = 0.0
    note: str = Field(default="", description="It the planner wants to communicate something important about this step it can do it here.")

class Plan(BaseModel):
    """The structured output of the planner agent, and the input of the executor agent."""
    reasoning: str = Field(..., description="The reasoning behind the plan, assumptions and information gathered from the tools.")
    steps: list[Step] = Field(..., description="The detailed step by step list of tool calls with exact, checked arguments.")

# Initialize the model via PydanticAI
planner_agent = Agent(
    model=RateLimitedModel(planner_model, rate_limiter),
    system_prompt=(
        "You are the planner part of a planner-executor agent to agent ecosystem. You two controll a Franka Emika Panda robot arm in Webots. You have no manipulation tools. You are responsible for generating a global plan to pick and place objects according to the instructions. Call get_objects (needs no arguments) and get_area_position only once. Call  get_target_position once for each object you need to move to gather information. Every pick and place step must set object_id, and every place step must set target_area."
        "The plan should be a detailed step by step list of tool calls with exact, correct arguments. The plan can only incorporate tools: 'pick(x: float, y: float, z: float, yaw: float = 0.0)', 'place(x: float, y: float, z: float, yaw: float = 0.0)', and 'go_home'. The plan must be clear and unambigous, because the executor agent will not be able to ask questions, this is an open loop implementation."
    ),
    toolsets=[planner_toolset],
    output_type=Plan,
    retries=4,
    model_settings= {'temperature' : 0.2,  'timeout':180, 'seed':42}
)
# Create plan
async def make_plan(task: str, usage:RunUsage) ->Plan:
    # The planner_agent evaluates the prompt, calls MCP tools as needed, and returns the result
    planner_result = await planner_agent.run(
        task,
        usage_limits=PLANNER_LIMITS,
        )
    usage.incr(planner_result.usage)
    print(f"\nPlanner agent > {planner_result.output}\n")
    return planner_result.output

# Initialize the model via PydanticAI
executor_agent = Agent(
    model=RateLimitedModel(executor_model, rate_limiter),
    system_prompt=(
        "You are the executor part of a planner-executor agent to agent ecosystem. You two controll a Franka Emika Panda robot arm in Webots."
        "Your goal is to pick and place objects according to the instructions. Use the provided tools to execute the steps, the plan contains. "
        "This is an open loop implementation.You don't ask questions, you just execute the plan provided by the planner agent. If you encounter an error stop and report the error."
    ),
    toolsets=[executor_toolset],
    model_settings= {'temperature' : 0, 'timeout':60,'max_tokens':512, 'seed':42}
)

# Execute the plan
async def execute_plan(plan: Plan, usage: RunUsage) -> str:
    # The executor_agent evaluates the plan, calls MCP tools as needed, and returns the result
    executor_result = await executor_agent.run(
                    plan.model_dump_json(indent=2),
                    usage_limits=EXECUTOR_LIMITS
                    )
    print(f"\nExecutor agent > {executor_result.output}\n")
    usage.incr(executor_result.usage)
    return executor_result.output

async def main():
    print("--- Panda Robot Webots Terminal CLI ---")
    print("Type 'exit' or 'quit' to end.\n")

    while True:
        user_input = input("Operator > ").strip()
        if not user_input or user_input.lower() in ["exit", "quit"]:
            break
        # Separate usage for both agents, fresh for each task
        planner_usage=RunUsage()
        executor_usage=RunUsage()
        # Create a plan
        plan= await make_plan(task=user_input, usage=planner_usage)
        # Execute the plan
        await execute_plan(plan=plan, usage=executor_usage)
        print(f"planner: {planner_usage}\nexecutor: {executor_usage}")
        

if __name__ == "__main__":
    asyncio.run(main())