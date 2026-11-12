"""Tenant-scope guards for Mongo reads (T0).

Every read in the message path must be scoped to one hotel. These helpers make
an unscoped query a loud error instead of a silent cross-tenant read.
"""

TENANT_KEYS = ("hotel_id", "hotelId", "HotelID")


def _has_tenant(node) -> bool:
    """True if a tenant key appears anywhere in the query, including nested
    inside $or / $and — the repos match hotel_id / hotelId / HotelID variants
    that way, so a top-level-only check would reject valid queries."""
    if isinstance(node, dict):
        return any(k in TENANT_KEYS for k in node) or any(_has_tenant(v) for v in node.values())
    if isinstance(node, list):
        return any(_has_tenant(v) for v in node)
    return False


def require_tenant_filter(query: dict) -> dict:
    """Return the query unchanged, or raise if it isn't hotel-scoped."""
    if not _has_tenant(query):
        raise ValueError(f"Unscoped tenant query refused: {query!r}")
    return query
