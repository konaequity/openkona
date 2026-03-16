#!/opt/anaconda3/bin/python3
"""KONASH Eval Server — unified server for Arena and Trace Viewer.

Run:
    python tools/server.py

Then open:
    http://localhost:5117/arena   — side-by-side model comparison
    http://localhost:5117/traces  — rollout trace analysis
"""
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.dirname(__file__))

from flask import Flask, redirect
from werkzeug.serving import run_simple

from arena.app import app as arena_app
from trace_viewer.app import app as trace_app

# Root app just redirects to /arena
root_app = Flask(__name__)


@root_app.route("/")
def home():
    return redirect("/training/")


# Combine via WSGI dispatch
class PathDispatcher:
    """Route requests to sub-apps based on URL prefix."""

    def __init__(self, default, mounts):
        self.default = default
        self.mounts = sorted(mounts.items(), key=lambda x: -len(x[0]))

    def __call__(self, environ, start_response):
        path = environ.get("PATH_INFO", "/")
        for prefix, app in self.mounts:
            if path == prefix or path.startswith(prefix + "/"):
                # Don't strip the prefix — routes already include it
                return app(environ, start_response)
        return self.default(environ, start_response)


app = PathDispatcher(root_app, {
    "/arena": arena_app,
    "/traces": trace_app,
    "/training": trace_app,
})

if __name__ == "__main__":
    port = int(os.environ.get("KONASH_PORT", 5050))
    print(f"\n  KONASH Eval Tools")
    print(f"  http://localhost:{port}\n")
    print(f"  /training — Training monitor")
    print(f"  /arena    — Model comparison arena")
    print(f"  /traces   — Rollout trace viewer\n")
    run_simple("0.0.0.0", port, app, use_debugger=True, use_reloader=True, threaded=True)
