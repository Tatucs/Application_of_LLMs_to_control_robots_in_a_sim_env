import asyncio
import logfire
from aiolimiter import AsyncLimiter
from pydantic_ai import Agent, UsageLimits, ModelMessage, RunUsage
from pydantic_ai.mcp import MCPToolset
from fastmcp.client.transports import StdioTransport

# Defining the Stdio connection to Webots MCP server script
# This script runs the FastMCP server, exposing the robot's capabilities as tools.
transport = StdioTransport(
    command="python",
    args=["src/llmbot/panda_mcp_server.py"]
)

# Wrap the transport in an MCPToolset
toolset = MCPToolset(transport)

# Initialize the Gemini model via PydanticAI
agent = Agent(
    'google:gemini-3.5-flash-lite',
    system_prompt=(
        "You are a robot arm operator controlling one Franka Emika Panda robot arm in Webots."
        "Use the provided tools to check gripper and object status, get target position, taret area or camera image. Your goal is to pick and place objects according to the instructions. "
        "Always check status or capabilities before executing complex arm movements."
        "Never guess placement coordinates, use values inside [x_min-x_max and [y_min-y_max from get_area_position()."
        "If pick returns grasp_failed: 1. Call get_objects again. 2. Call get_target_position again for that cube’s current ID. 3. Only retry if it is visible and a fresh position was returned. 4. Otherwise call get_camera_image and report that the cube is not visible."
    ),
    toolsets=[toolset],
    model_settings= {'temperature' : 0.2, 'max_tokens':512, 'seed':42, 'stop_sequences':["\\"]}
)

logfire.configure(send_to_logfire='if-token-present')
logfire.instrument_pydantic_ai()

rate_limiter = AsyncLimiter(max_rate=15, time_period=61)

usage_limits=UsageLimits(request_limit=10)
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
            
            # The agent evaluates the prompt, calls MCP tools as needed, and returns the result
            result = await agent.run(
                user_input,
                usage=usage,
                message_history=message_history,
                #usage_limits=usage_limits,
                )
            print(f"\nRobot > {result.output}\n")
            print(f"Usage details: {result.usage}\n")

if __name__ == "__main__":
    asyncio.run(main())