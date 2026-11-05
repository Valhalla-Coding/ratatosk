import ipaddress
from dataclasses import dataclass, field

from ratatosk.gpuqueue import P_IMAGE, P_LLM, P_RP
from ratatosk.settings import Route, Settings

DROP_BASE = -1000  # a "drop everything" line outranks every normal line


@dataclass(frozen=True)
class Rank:
    priority: int  # lower goes first
    preempt: bool  # cancel the running job when this request arrives
    label: str  # what the request log shows as the reason
    door: int | None = field(default=None, compare=False)  # the port the request came in on; lets the UI show which door is busy


def _matches(line: dict, route: Route, ip: str | None, agent: str) -> bool:
    if "door" in line:
        return line["door"] == route.port
    try:
        inside = ipaddress.ip_address(ip) in ipaddress.ip_network(line["client"], strict=False)
    except (ValueError, TypeError):
        return False
    return inside and line.get("agent", "").lower() in agent.lower()


def rank_for(rules: list[dict], route: Route, ip: str | None, agent: str) -> Rank:
    """The first matching line decides; a request no line matches waits in turn after all of them."""
    for i, line in enumerate(rules):
        if _matches(line, route, ip, agent):
            label = line.get("label") or (f"door {line['door']}" if "door" in line else line["client"])
            if line.get("drop"):
                return Rank(DROP_BASE + i, True, label, route.port)
            return Rank(i, False, label, route.port)
    return Rank(len(rules), False, "Everything else", route.port)


def rank_request(settings: Settings, route: Route, ip: str | None, agent: str) -> Rank:
    """Rank for one request. Without a priority list the order is the one Ratatosk always had: roleplay, text, images."""
    if settings.priority is None:
        base = P_RP if route.kind == "llm" and route.profile == "rp" else P_LLM if route.kind == "llm" else P_IMAGE
        return Rank(base, False, f"door {route.port}", route.port)
    return rank_for(settings.priority, route, ip, agent)
