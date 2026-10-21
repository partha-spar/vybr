"""Standalone Flask entrypoint for the 'geo' microservice."""
import os

os.environ["SERVICE_NAME"] = "geo"

from app import app  # noqa: E402,F401

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
