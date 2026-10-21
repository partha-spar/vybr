"""Standalone Flask entrypoint for the 'auth' microservice."""
import os

os.environ["SERVICE_NAME"] = "auth"

from app import app  # noqa: E402,F401

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
