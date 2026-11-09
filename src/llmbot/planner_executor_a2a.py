import asyncio
import sys
from pathlib import Path
import logfire
from aiolimiter import AsyncLimiter
from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelMessage, RunUsage
from pydantic_ai.mcp import MCPToolset
from fastmcp.client.transports import StdioTransport
from llmbot.rate_limit import RateLimitedModel

# Defining the Stdio connection to Webots MCP server script
# This script runs the FastMCP server, exposing the robot's capabilities as tools.
transport = StdioTransport(
    command=sys.executable,
    args=[str(Path(__file__).resolve().parent / "panda_mcp_server.py")]
)

logfire.configure(send_to_logfire='if-token-present')
logfire.instrument_pydantic_ai()


model2='openai-chat:Qwen/Qwen3.8-27B-NVFP4'
model1='google:gemini-3.5-flash-lite'

# Wrap the transport in an MCPToolset
toolset = MCPToolset(transport)

rate_limiter = AsyncLimiter(max_rate=1, time_period=5)
# Creating filtered toolset for the agents

PLANNER_TOOLS = {"get_area_position","get_camera_image","get_gripper_state","get_objects","get_target_position"}
EXECUTOR_TOOLS = {"pick","place","go_home"}
planner_toolset = toolset.filtered(lambda ctx, tool: tool.name in PLANNER_TOOLS)
executor_toolset = toolset.filtered(lambda ctx, tool: tool.name in EXECUTOR_TOOLS)

# Define output type for planner agent
class PlannerOutput(BaseModel):
    """The JSON output of the planner agent, and the input of the executor agent."""
    reasoning: str = Field(..., description="The reasoning behind the plan, assumptions and information gathered from the tools.")
    full_plan: list[dict] = Field(..., description="The detailed step by step list of tool calls with exact, checked arguments.")

# Initialize the model via PydanticAI
planner_agent = Agent(
    model=RateLimitedModel(model2, rate_limiter),
    system_prompt=(
        "You are the planner part of a planner-executor agent to agent ecosystem. You two controll a Franka Emika Panda robot arm in Webots. You have no manipulation tools. You are responsible for generating a global plan to pick and place objects according to the instructions. Run each of your tools only once to gather information, then write the plan!"
        "Your output must be a single JSON object with keys: 'reasoning' and 'full plan'. The plan should be a detailed step by step list of tool calls with exact, correct arguments. The plan can only incorporate tools: 'pick(x: float, y: float, z: float, yaw: float = 0.0)', 'place(x: float, y: float, z: float, yaw: float = 0.0)', and 'go_home'. The plan must be clear and unambigous, because the executor agent will not be able to ask questions, this is an open loop implementation."
    ),
    toolsets=[planner_toolset],
    output_type=PlannerOutput,
    model_settings= {'temperature' : 0.2, 'seed':42}
)
# Create plan
async def create_plan(task: str, planner_agent: Agent[object, PlannerOutput]) ->str:
    # The planner_agent evaluates the prompt, calls MCP tools as needed, and returns the result
    planner_result = await planner_agent.run(
        task,
        usage=usage,
        message_history=message_history,
        )
    print(f"\nPlanner agent > {planner_result.output}\n")
    print(f"Usage details of planner agent: {planner_result.usage}\n")
    plan: PlannerOutput=planner_result.output
    return plan.model_dump_json(indent=2)

# Initialize the model via PydanticAI
executor_agent = Agent(
    model=RateLimitedModel(model1, rate_limiter),
    system_prompt=(
        "You are the executor part of a planner-executor agent to agent ecosystem. You two controll a Franka Emika Panda robot arm in Webots."
        "Your goal is to pick and place objects according to the instructions. Use the provided tools to execute the steps, the plan contains. "
        "This is an open loop implementation.You don't ask questions, you just execute the plan provided by the planner agent. If you encounter an error stop try to recover and continue, if you can't recover stop and report the error."
    ),
    toolsets=[executor_toolset],
    model_settings= {'temperature' : 0, 'max_tokens':512, 'seed':42}
)

# Execute the plan
async def execute_plan(plan: str, executor_agent: Agent):
    # The executor_agent evaluates the plan, calls MCP tools as needed, and returns the result
    executor_result = await executor_agent.run(
                    plan,
                    usage=usage,
                    message_history=message_history,
                    )
    print(f"\nExecutor agent > {executor_result.output}\n")
    print(f"Usage details of executor agent: {executor_result.usage}\n")

rate_limiter = AsyncLimiter(max_rate=15, time_period=65)

message_history:list[ModelMessage] | None=None
usage: RunUsage=RunUsage()

async def main():
    print("--- Panda Robot Webots Terminal CLI ---")
    print("Type 'exit' or 'quit' to end.\n")
        
    while True:
        user_input = input("Operator > ").strip()
        if not user_input or user_input.lower() in ["exit", "quit"]:
            break
        # Create a plan    
        plan= await create_plan(task=user_input, planner_agent=planner_agent)
        # Execute the plan
        await execute_plan(plan=plan,executor_agent=executor_agent)
        

if __name__ == "__main__":
    asyncio.run(main())