"""URL path helpers for outbound HTTP.

Re-exported from the shared runner contract so both framework adapters use
one traversal-defense choke point.
"""

from agent_engine_runner_shared.http_path import quote_path_segment

__all__ = ["quote_path_segment"]
