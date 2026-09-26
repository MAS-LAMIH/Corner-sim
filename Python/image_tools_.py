"""Backward-compatible alias for the maintained :mod:`image_tools` module.

Historically this file contained a divergent copy of the annotation pipeline.  It is
kept as an import alias so external research scripts continue to work without
preserving a second, potentially incorrect implementation.
"""

from image_tools import *  # noqa: F401,F403
