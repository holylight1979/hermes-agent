"""Registration half of the llm-cr prompt injection.

All behaviour lives in :mod:`injector` (an explicit script module in this plugin directory) so the
load + gate + inject logic is testable without the plugin loader. This file only wires the
``llm_execution`` middleware, which is the LAST stage before the provider call
(``agent/turn_api_call.py`` -> ``run_llm_execution_middleware``): request hooks, request middleware
and debug dumps have all already run on the un-injected payload, and the injected text exists only
in the copy handed to the transport.

The sibling import is deliberately inside ``register()``: the plugin loader gives this directory a
package context, but a plain ``import`` of this file (pytest collecting the directory alongside the
plugin's own tests) does not — at module level a relative import would hard-fail there.
"""

from __future__ import annotations


def register(ctx):
    from hermes_cli.middleware import LLM_EXECUTION_MIDDLEWARE

    from .injector import make_middleware

    ctx.register_middleware(LLM_EXECUTION_MIDDLEWARE, make_middleware(ctx))
