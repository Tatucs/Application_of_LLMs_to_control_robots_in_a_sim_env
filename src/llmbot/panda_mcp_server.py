'''
MCP server fopr controlling the Panda robot in Webots.

This server works in conjkunction with the Webots controller (panda_mcp_controller.py)
via the file system and sockets to exchange commands and status.
'''

import json
import logging
import os
import sys
import time
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

# ----------------------------------------- FILE I/O
def write_json(path: Path, data: dict, retries: int = 20) -> None:
    """Write data to a JSON file atomically.

    Writes to a .tmp file first, then swaps it in with os.replace, so the
    controller never reads a half-written file. On Windows os.replace fails
    while the controller has the file open, so the swap is retried.

    Args:
        path: Target file, e.g. COMMANDS_FILE.
        data: JSON-serialisable dict to write.
        retries: Swap attempts before the last one, 10 ms apart.

    Returns:
        None.

    Raises:
        PermissionError: If the file is still locked after all retries.
        TypeError: If data is not JSON-serialisable.
    """
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data), encoding="utf-8")
    for _ in range(retries):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:   # Windows: controller has commands.json open right now
            time.sleep(0.01)
    os.replace(tmp, path)         # last try: real failure -> raise

def read_json(path: Path) -> dict | None:
    """Read and parse a JSON file without raising.

    Args:
        path: File to read, e.g. STATUS_FILE.

    Returns:
        The parsed content, or None if the file is missing, half-written
        or locked. None means "not available yet": the caller should
        poll again.
    """
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, PermissionError, json.JSONDecodeError):
        return None


# ----------------------------------------- main
def main() -> None:
    """Create the data directory and start the MCP server on stdio.

    Blocks until the client closes the connection.
    """
    logger.info(f"Data directory: {DATA_DIR}")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Starting Panda MCP Server...")
    mcp.run()


if __name__ == "__main__":
    main()