"""Start the web UI: ``python -m app.ui`` (Streamlit on ``UI_HOST``:``UI_PORT``, default 127.0.0.1:8501).

The launcher fixes the Streamlit options that matter for a deployment:

- headless, with no usage statistics sent to Streamlit and no public-IP lookup
  (``UI_PUBLIC_ADDRESS`` names the address users browse to);
- no file watcher and no uploads;
- XSRF protection on.

With ``APP_ENV=production`` it also shows no exception details in the browser and only the
viewer toolbar. Tracebacks go to the server log, never to users.
"""

from __future__ import annotations

import sys
from pathlib import Path

from streamlit.web import cli as streamlit_cli

from app.config import get_settings, settings_or_exit

PAGE = Path(__file__).with_name("main.py")


def streamlit_args() -> list[str]:
    settings = get_settings()
    production = settings.app_env == "production"
    return [
        "streamlit",
        "run",
        str(PAGE),
        "--server.headless=true",
        f"--server.address={settings.ui_host}",
        f"--server.port={settings.ui_port}",
        # Set explicitly, so Streamlit never looks up the host's public IP address at start-up.
        f"--browser.serverAddress={settings.ui_public_address}",
        "--server.enableXsrfProtection=true",
        "--server.fileWatcherType=none",
        "--server.maxUploadSize=1",
        "--browser.gatherUsageStats=false",
        f"--client.showErrorDetails={'none' if production else 'full'}",
        f"--client.toolbarMode={'viewer' if production else 'auto'}",
    ]


def main() -> None:
    settings_or_exit()  # invalid settings: a list of the settings to fix, exit code 2
    sys.argv = streamlit_args()
    sys.exit(streamlit_cli.main())


if __name__ == "__main__":
    main()
