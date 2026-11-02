import asyncio
import sys
from pathlib import Path
import logfire
from aiolimiter import AsyncLimiter
from pydantic_ai import Agent, UsageLimits, ModelMessage, RunUsage
from pydantic_ai.mcp import MCPToolset
from fastmcp.client.transports import StdioTransport
#from fasta2a.client import A2AClient
#from fasta2a.schema import Message
#import nest_asyncio

# Defining the Stdio connection to Webots MCP server script
# This script runs the FastMCP server, exposing the robot's capabilities as tools.
transport = StdioTransport(
    command=sys.executable,
    args=[str(Path(__file__).resolve().parent / "panda_mcp_server.py")]
)

logfire.configure(send_to_logfire='if-token-present')
logfire.instrument_pydantic_ai()

#nest_asyncio.apply()

model2='openai-chat:Qwen/Qwen3.8-27B-NVFP4'
model1='google:gemini-3.6-flash'

# Wrap the transport in an MCPToolset
toolset = MCPToolset(transport)

# Initialize the MIT Gwen model via PydanticAI
planner_agent = Agent(
    model=model1,
    system_prompt=(
        "You are the planner part of a planner-executor agent to agent ecosystem. You two controll a Franka Emika Panda robot arm in Webots. You are responsible for generating a global plan to pick and place objects according to the instructions. First gather information. Incorporate in the plan the provided tools to check gripper and object status, get target position, target area or camera image."
        "The plan should be a detailed step by step list of actions to be executed by the executor agent, it must be clear and unambigous, because the executor agent will not be able to ask questions."
    ),
    toolsets=[toolset],
    model_settings= {'temperature' : 0.2, 'max_tokens':512, 'seed':42, 'stop_sequences':["\\"]}
)


# Initialize the Gemini model via PydanticAI
executor_agent = Agent(
    model=model2,
    system_prompt=(
        "You are the executor part of a planner-executor agent to agent ecosystem. You two controll a Franka Emika Panda robot arm in Webots."
        "Use the provided tools to check gripper and object status, get target position, taret area or camera image. Your goal is to pick and place objects according to the instructions. "
        "You don't ask questions, you just execute the plan provided by the planner agent. If you encounter an error stop try to recover and continue, if you can't recover stop and report the error. "
        "Never guess placement coordinates, use values inside [x_min-x_max and [y_min-y_max from get_area_position()."
        "If pick returns grasp_failed: 1. Call get_objects again. 2. Call get_target_position again for that cube’s current ID. 3. Only retry if it is visible and a fresh position was returned. 4. Otherwise call get_camera_image and report that the cube is not visible."
        "Place object only on empty positions within target area!"
    ),
    toolsets=[toolset],
    model_settings= {'temperature' : 0.2, 'max_tokens':1000, 'seed':42, 'stop_sequences':["\\"]}
)

rate_limiter = AsyncLimiter(max_rate=15, time_period=65)

message_history:list[ModelMessage] | None=None
usage: RunUsage=RunUsage()

async def main():
    # The 'async with' block automatically starts the stdio subprocess and loads the tools
    async with rate_limiter:
        print("--- Panda Robot Webots Terminal CLI ---")
        print("Type 'exit' or 'quit' to end.\n")
        
        while True:
            user_input = input("Operator > ").strip()
            if not user_input or user_input.lower() in ["exit", "quit"]:
                break
            
            # The planner_agent evaluates the prompt, calls MCP tools as needed, and returns the result
            planner_result = await planner_agent.run(
                user_input,
                usage=usage,
                message_history=message_history,
                )
            print(f"\nPlanner agent > {planner_result.output}\n")
            # The executor_agent evaluates the plan, calls MCP tools as needed, and returns the result
            executor_result = await executor_agent.run(
                            planner_result.output,
                            usage=usage,
                            message_history=message_history,
                            )
            print(f"\nExecutor agent > {executor_result.output}\n")
            print(f"Usage details of planner agent: {planner_result.usage}\n")
            print(f"Usage details of executor agent: {executor_result.usage}\n")

if __name__ == "__main__":
    asyncio.run(main())