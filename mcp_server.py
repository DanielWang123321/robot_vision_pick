"""MCP tool server for Vision Pick system.

Exposes scan_table, pick_object, and get_status as MCP tools
for integration with OpenClaw and other MCP-compatible agents.

Usage:
    python mcp_server.py

Environment variables:
    VISION_PICK_CONFIG   - Path to config file (default: config.yaml)
    VISION_PICK_DRY_RUN  - Set to "1" to skip real robot movement
"""

import json
import logging
import os
import sys
from contextlib import asynccontextmanager

import yaml

# Force all logging to stderr (stdout is reserved for MCP protocol)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stderr,
)

from main import validate_config
from pick_system import VisionPickSystem

from mcp.server.fastmcp import FastMCP, Context

logger = logging.getLogger(__name__)


def _load_config():
    path = os.environ.get("VISION_PICK_CONFIG", "config.yaml")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@asynccontextmanager
async def lifespan(server: FastMCP):
    cfg = _load_config()
    validate_config(cfg)

    dry_run = os.environ.get("VISION_PICK_DRY_RUN", "").lower() in ("1", "true", "yes")
    system = VisionPickSystem(cfg, dry_run=dry_run)

    state = {
        "system": system,
        "dry_run": dry_run,
        "total_picked": 0,
        "last_scan": None,
        "last_assessments": None,
    }

    try:
        logger.info("Vision Pick MCP server ready (dry_run=%s)", dry_run)
        yield state
    finally:
        logger.info("Shutting down Vision Pick system")
        system.cleanup()


mcp = FastMCP("vision-pick", lifespan=lifespan)


@mcp.tool()
def scan_table(ctx: Context) -> str:
    """Scan the table and return detected objects with scores.

    Moves the robot to the detection pose, captures an RGB+depth image,
    runs LLM detection, and returns a ranked list of graspable objects.
    Call this before pick_object to get available targets.
    """
    state = ctx.request_context.lifespan_context
    system = state["system"]

    scan_result = system.scan_objects()

    if not scan_result.objects:
        state["last_scan"] = None
        state["last_assessments"] = None
        return json.dumps({"objects": [], "count": 0})

    assessments = system.assess_targets(scan_result)

    state["last_scan"] = scan_result
    state["last_assessments"] = assessments

    objects = []
    for a in assessments:
        obj = a.detected_object
        objects.append({
            "target_id": obj.display_id,
            "name": obj.name,
            "category": obj.category,
            "score": round(a.total_score, 1),
            "confidence": round(obj.confidence, 2),
            "graspable": obj.graspable,
            "hint": obj.grasp_reason,
        })

    return json.dumps({"objects": objects, "count": len(objects)})


@mcp.tool()
def pick_object(target_id: str, ctx: Context) -> str:
    """Pick a specific object from the table by its target ID (e.g. "T1").

    Must call scan_table first to get available targets.
    After picking, the scan cache is cleared -- call scan_table again for the next pick.
    """
    state = ctx.request_context.lifespan_context
    system = state["system"]
    scan_result = state.get("last_scan")
    assessments = state.get("last_assessments")

    if scan_result is None or assessments is None:
        return json.dumps({
            "success": False,
            "reason": "no_scan",
            "object_name": None,
            "verified": None,
        })

    target_key = target_id.strip().upper()
    selected = None
    for a in assessments:
        if a.detected_object.display_id.upper() == target_key:
            selected = a
            break

    if selected is None:
        available = [a.detected_object.display_id for a in assessments]
        return json.dumps({
            "success": False,
            "reason": "target_not_found",
            "object_name": None,
            "verified": None,
            "available_targets": available,
        })

    detected_obj = selected.detected_object
    pick_result = system.pick_object(detected_obj, scan_result, selected_assessment=selected)

    # Clear cache -- table state changed
    state["last_scan"] = None
    state["last_assessments"] = None

    if pick_result.success:
        state["total_picked"] += 1

    return json.dumps({
        "success": pick_result.success,
        "reason": pick_result.reason,
        "object_name": detected_obj.name,
        "verified": pick_result.verified,
    })


@mcp.tool()
def get_status(ctx: Context) -> str:
    """Get current system status: connection, pick count, and scan state."""
    state = ctx.request_context.lifespan_context
    return json.dumps({
        "robot_connected": not state["dry_run"],
        "dry_run": state["dry_run"],
        "total_picked": state["total_picked"],
        "has_cached_scan": state.get("last_scan") is not None,
    })


if __name__ == "__main__":
    mcp.run()
