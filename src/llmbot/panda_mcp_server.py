'''
MCP server fopr controlling the Panda robot in Webots.

This server works in conjkunction with the Webots controller (panda_mcp_controller.py)
via the file system and sockets to exchange commands and status.
'''

import json
import os
import logging
import sys
from pathlib import Path
from fastmcp import FastMCP

# Configure logging
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filemode='a',
    encoding='utf-8'
)
logger = logging.getLogger(__name__)

# Define paths for data exchange with the controller
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "panda"
COMMANDS_FILE = DATA_DIR / "commands.json"
STATUS_FILE = DATA_DIR / "status.json"

# Create MCP server
mcp = FastMCP("Panda MCP Server")


# ----------------------------------------- main
def main():
    logger.info(f"Data directory: {DATA_DIR}")
    logger.info("Starting Panda MCP Server...")
    mcp.run()


if __name__ == "__main__":
    main()