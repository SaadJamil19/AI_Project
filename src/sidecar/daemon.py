from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import socketserver
import struct
import sys
import threading
import time
from pathlib import Path
from typing import Any

from pydantic import ValidationError

try:
    from .search import SearchService, build_service
except ImportError:  # pragma: no cover - supports direct script execution during local debugging
    from search import SearchService, build_service  # type: ignore[no-redef]


MAX_REQUEST_BYTES = 64 * 1024
DEFAULT_SOCKET_NAME = "sidecar.sock"


def default_runtime_dir() -> Path:
    if value := os.environ.get("SEMANTIC_CLI_AGENT_HOME"):
        return Path(value).expanduser()
    if value := os.environ.get("XDG_DATA_HOME"):
        return Path(value).expanduser() / "semantic-cli-agent"
    return Path.home() / ".local" / "share" / "semantic-cli-agent"


def ensure_private_runtime_dir(path: Path) -> None:
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    if os.name == "posix":
        os.chmod(path, 0o700)
        mode = path.stat().st_mode & 0o777
        if mode != 0o700:
            raise PermissionError(f"{path} has mode {mode:o}; expected 700")


def peer_uid(conn: socket.socket) -> int:
    if not hasattr(socket, "SO_PEERCRED"):
        raise RuntimeError("SO_PEERCRED is not available on this platform")
    raw = conn.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i"))
    _pid, uid, _gid = struct.unpack("3i", raw)
    return int(uid)


class SidecarUnixServer(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True
    allow_reuse_address = False

    def __init__(self, socket_path: Path, service: SearchService) -> None:
        self.socket_path = socket_path
        self.service = service
        self.expected_uid = os.getuid()
        self._shutdown_lock = threading.Lock()
        if socket_path.exists():
            socket_path.unlink()
        super().__init__(str(socket_path), SidecarRequestHandler)
        os.chmod(socket_path, 0o600)

    def server_close(self) -> None:
        with self._shutdown_lock:
            super().server_close()
            try:
                if self.socket_path.exists():
                    self.socket_path.unlink()
            except FileNotFoundError:
                pass


class SidecarRequestHandler(socketserver.StreamRequestHandler):
    server: SidecarUnixServer

    def handle(self) -> None:
        handler_started_at = time.perf_counter()
        try:
            uid = peer_uid(self.connection)
            if uid != self.server.expected_uid:
                self._write_error("peer_uid_mismatch", f"uid {uid} is not authorized")
                return

            payload = self.rfile.readline(MAX_REQUEST_BYTES + 1)
            if not payload:
                self._write_error("empty_request", "request payload is empty")
                return
            if len(payload) > MAX_REQUEST_BYTES:
                self._write_error("payload_too_large", "request exceeds 64KiB")
                return

            stripped = payload.strip()
            if self._is_invalidation_command(stripped):
                response = self.server.service.handle_invalidate_json(stripped)
            else:
                response = self.server.service.handle_json(
                    stripped,
                    dequeued_at=handler_started_at,
                )
            self.wfile.write(response)
        except ValidationError as exc:
            self._write_error("validation_error", exc.json())
        except Exception as exc:  # Deliberate boundary: never crash worker thread on bad client input.
            self._write_error("internal_error", str(exc))

    @staticmethod
    def _is_invalidation_command(payload: bytes) -> bool:
        try:
            envelope = json.loads(payload)
        except json.JSONDecodeError:
            return False
        return isinstance(envelope, dict) and envelope.get("command") == "invalidate_cache"

    def _write_error(self, code: str, message: str) -> None:
        safe_message = message.replace("\n", " ")[:4096]
        payload = json.dumps(
            {
                "protocol_version": "1.0.0",
                "source_provenance": "LOCAL_ML_SIDECAR",
                "error": {"code": code, "message": safe_message},
            },
            separators=(",", ":"),
        )
        self.wfile.write(payload.encode("utf-8") + b"\n")


def install_signal_handlers(server: SidecarUnixServer) -> None:
    def stop(_signum: int, _frame: Any) -> None:
        server.shutdown()
        server.server_close()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="semantic-cli-agent Python ML sidecar")
    parser.add_argument("--runtime-dir", type=Path, default=default_runtime_dir())
    parser.add_argument("--db-path", type=Path, default=None)
    parser.add_argument("--socket-path", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    runtime_dir: Path = args.runtime_dir.expanduser().resolve()
    ensure_private_runtime_dir(runtime_dir)

    db_path: Path = (
        args.db_path.expanduser().resolve()
        if args.db_path is not None
        else runtime_dir / "cli-agent.db"
    )
    socket_path: Path = (
        args.socket_path.expanduser().resolve()
        if args.socket_path is not None
        else runtime_dir / DEFAULT_SOCKET_NAME
    )

    service = build_service(db_path)
    server = SidecarUnixServer(socket_path=socket_path, service=service)
    install_signal_handlers(server)

    try:
        print(f"semantic-cli-agent sidecar listening on {socket_path}", flush=True)
        server.serve_forever(poll_interval=0.5)
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
