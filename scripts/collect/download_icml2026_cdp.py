"""Rate-limited ICML 2026 PDF download with OpenReview CDP challenge refresh.

Public network requests still go through the configured proxy. This helper uses
an already-open Chrome remote debugging session only to refresh/export the
short-lived OpenReview clearance cookie when needed.

Usage:
    python scripts/collect/download_icml2026_cdp.py --limit 20
    python scripts/collect/download_icml2026_cdp.py --batch-size 20
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.parse
import base64
from pathlib import Path
from typing import Any

import requests
import websocket

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from config import settings
from pipeline.paper_crawler.ccf_a_main_corpus import (
    CorpusPaper,
    MainConferenceCorpusBuilder,
)


COOKIE_FILE = Path(os.environ.get("TEMP", ".")) / "retrival-openreview-cdp-cookies.json"
CDP_BASE = "http://127.0.0.1:9222"
VENUE = "ICML"
YEAR = 2026
CHROME_BOUNDS = {"left": 10, "top": 10, "width": 1051, "height": 798, "windowState": "normal"}
MIN_INTERVAL_FLOOR_SECONDS = 5.0


class Cdp:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.ws = None
        self._id = 0
        self.session = requests.Session()
        self.session.trust_env = False
        self.window_id: int | None = None

    def __enter__(self) -> "Cdp":
        targets = self.session.get(f"{self.base_url}/json", timeout=5).json()
        target = next(
            (t for t in targets if t.get("type") == "page" and "openreview" in (t.get("url") or "")),
            next(t for t in targets if t.get("type") == "page"),
        )
        self.ws = websocket.create_connection(
            target["webSocketDebuggerUrl"],
            timeout=10,
            suppress_origin=True,
        )
        self.call("Page.enable")
        self.call("Runtime.enable")
        self.call("Network.enable")
        self.normalize_window()
        return self

    def __exit__(self, *_exc: object) -> None:
        if self.ws is not None:
            self.ws.close()

    def call(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        if self.ws is None:
            raise RuntimeError("CDP connection is not open")
        self._id += 1
        message_id = self._id
        self.ws.send(json.dumps({"id": message_id, "method": method, "params": params or {}}))
        while True:
            message = json.loads(self.ws.recv())
            if message.get("id") == message_id:
                if "error" in message:
                    raise RuntimeError(f"CDP {method} failed: {message['error']}")
                return message.get("result", {})

    def eval(self, expression: str) -> Any:
        result = self.call("Runtime.evaluate", {"expression": expression, "returnByValue": True})
        return result.get("result", {}).get("value")

    def normalize_window(self) -> dict[str, Any]:
        info = self.call("Browser.getWindowForTarget")
        self.window_id = int(info["windowId"])
        bounds = info.get("bounds", {})
        needs_reset = (
            bounds.get("windowState") != "normal"
            or int(bounds.get("left", 0)) < 0
            or int(bounds.get("top", 0)) < 0
            or int(bounds.get("width", 0)) < 900
            or int(bounds.get("height", 0)) < 650
        )
        if needs_reset:
            self.call(
                "Browser.setWindowBounds",
                {"windowId": self.window_id, "bounds": CHROME_BOUNDS},
            )
            time.sleep(0.5)
            bounds = self.call("Browser.getWindowForTarget").get("bounds", {})
        return bounds

    def cookies(self) -> list[dict[str, Any]]:
        cookies = [
            cookie
            for cookie in self.call("Network.getAllCookies").get("cookies", [])
            if "openreview.net" in cookie.get("domain", "")
        ]
        COOKIE_FILE.write_text(json.dumps(cookies, ensure_ascii=False, indent=2), encoding="utf-8")
        return cookies

    def screenshot(self, path: Path) -> None:
        encoded = self.call("Page.captureScreenshot", {"format": "png", "fromSurface": True})["data"]
        path.write_bytes(base64.b64decode(encoded))


def clearance_seconds_left(cookies: list[dict[str, Any]]) -> float:
    clearance = next((cookie for cookie in cookies if cookie.get("name") == "openreview.clearanceToken"), None)
    if clearance is None:
        return 0.0
    return float(clearance.get("expires") or 0.0) - time.time()


def click_screen(x: int, y: int) -> None:
    # Use a Win32 click from PowerShell. It is a real OS mouse click against the
    # visible Chrome window, not a synthetic DOM click.
    ps_script = f"""
Add-Type @"
using System;
using System.Runtime.InteropServices;
public class NativeClick {{
  [DllImport("user32.dll")] public static extern bool SetCursorPos(int X, int Y);
  [DllImport("user32.dll")] public static extern void mouse_event(uint dwFlags, uint dx, uint dy, uint dwData, UIntPtr dwExtraInfo);
  [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr hWnd);
}}
"@
$proc = Get-Process chrome | Where-Object {{ $_.MainWindowTitle -like '*OpenReview*' -or $_.MainWindowTitle -like '*Verifying*' }} | Select-Object -First 1
if ($proc) {{ [NativeClick]::SetForegroundWindow($proc.MainWindowHandle) | Out-Null; Start-Sleep -Milliseconds 250 }}
[NativeClick]::SetCursorPos({x}, {y}) | Out-Null
Start-Sleep -Milliseconds 180
[NativeClick]::mouse_event(0x0002, 0, 0, 0, [UIntPtr]::Zero)
Start-Sleep -Milliseconds 100
[NativeClick]::mouse_event(0x0004, 0, 0, 0, [UIntPtr]::Zero)
"""
    subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps_script],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def refresh_clearance(
    cdp_base: str,
    forum_id: str,
    min_seconds_left: float,
    *,
    settle_seconds: float,
) -> float:
    redirect = f"https://openreview.net/pdf?id={forum_id}"
    challenge_url = "https://openreview.net/challenge?redirect=" + urllib.parse.quote(redirect, safe="")
    with Cdp(cdp_base) as cdp:
        bounds = cdp.normalize_window()
        cdp.call("Page.navigate", {"url": challenge_url})
        time.sleep(max(1.0, settle_seconds))
        bounds = cdp.normalize_window()

        # Browser/page geometry is in Windows logical pixels on this host. Use
        # Browser bounds instead of JS screenX/screenY; Chromium returns invalid
        # screen coordinates when a window was minimized or restored oddly.
        geom = cdp.eval(
            "({outerWidth,outerHeight,innerWidth,innerHeight,devicePixelRatio})"
        )
        window_left = float(bounds.get("left", CHROME_BOUNDS["left"]))
        window_top = float(bounds.get("top", CHROME_BOUNDS["top"]))
        window_width = float(bounds.get("width", geom["outerWidth"]))
        window_height = float(bounds.get("height", geom["outerHeight"]))
        content_left = window_left + (window_width - float(geom["innerWidth"])) / 2.0
        content_top = window_top + (window_height - float(geom["innerHeight"]))
        checkbox_x = int(content_left + float(geom["innerWidth"]) / 2.0 - 130)
        checkbox_y = int(content_top + float(geom["innerHeight"]) / 2.0 + 5)
        points = [
            (checkbox_x, checkbox_y),
            (checkbox_x - 9, checkbox_y),
            (checkbox_x + 9, checkbox_y),
            (checkbox_x, checkbox_y - 9),
            (checkbox_x, checkbox_y + 9),
        ]

        for attempt in range(1, 16):
            x, y = points[(attempt - 1) % len(points)]
            click_screen(x, y)
            time.sleep(2.0)
            seconds_left = clearance_seconds_left(cdp.cookies())
            page_url = cdp.eval("location.href")
            print(
                f"[{time.strftime('%H:%M:%S')}] challenge click {attempt}: "
                f"point=({x},{y}) seconds_left={seconds_left:.0f} url={page_url}",
                flush=True,
            )
            if seconds_left >= min_seconds_left:
                return seconds_left
        shot = Path(os.environ.get("TEMP", ".")) / "retrival-openreview-refresh-failed.png"
        cdp.screenshot(shot)
        print(f"[{time.strftime('%H:%M:%S')}] challenge refresh failed; screenshot={shot}", flush=True)
        return clearance_seconds_left(cdp.cookies())


def refresh_usable_clearance(
    cdp_base: str,
    forum_id: str,
    *,
    safety_seconds: float,
    min_interval_seconds: float,
    settle_seconds: float,
    retry_seconds: float,
) -> float:
    min_seconds_left = safety_seconds + min_interval_seconds
    while True:
        try:
            seconds_left = refresh_clearance(
                cdp_base,
                forum_id,
                min_seconds_left,
                settle_seconds=settle_seconds,
            )
        except (
            requests.RequestException,
            websocket.WebSocketException,
            OSError,
            RuntimeError,
        ) as exc:
            print(
                f"[{time.strftime('%H:%M:%S')}] clearance refresh failed "
                f"({type(exc).__name__}: {exc}); retrying in {retry_seconds:.0f}s",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(max(1.0, retry_seconds))
            continue

        if seconds_left >= min_seconds_left:
            return seconds_left

        print(
            f"[{time.strftime('%H:%M:%S')}] clearance not usable "
            f"(seconds_left={seconds_left:.0f}); retrying in {retry_seconds:.0f}s",
            file=sys.stderr,
            flush=True,
        )
        time.sleep(max(1.0, retry_seconds))


def pdf_path_for(builder: MainConferenceCorpusBuilder, paper: CorpusPaper) -> Path:
    return builder.pdf_root / paper.conference.lower() / str(paper.year) / f"{paper.paper_id}.pdf"


def _scan_json_depth(line: str, *, depth: int, in_string: bool, escape: bool) -> tuple[int, bool, bool]:
    for ch in line:
        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
    return depth, in_string, escape


def iter_icml2026_manifest_records(path: Path):
    """Yield ICML 2026 records from the pretty manifest without parsing all rows."""
    needle = '"paper_id": "icml-2026-'
    size = path.stat().st_size
    window = 8 * 1024 * 1024
    while True:
        offset = max(0, size - window)
        with path.open("rb") as f:
            f.seek(offset)
            text = f.read().decode("utf-8", errors="replace")
        first_hit = text.find(needle)
        if first_hit >= 1024 * 1024 or offset == 0:
            break
        window *= 2

    previous_line = ""
    collecting = False
    depth = 0
    in_string = False
    escape = False
    buf: list[str] = []

    for line in text.splitlines(keepends=True):
        if not collecting:
            if needle not in line:
                previous_line = line
                continue

            buf = []
            depth = 0
            in_string = False
            escape = False
            if previous_line.lstrip().startswith("{"):
                buf.append(previous_line)
                depth, in_string, escape = _scan_json_depth(
                    previous_line, depth=depth, in_string=in_string, escape=escape
                )
            collecting = True

        buf.append(line)
        depth, in_string, escape = _scan_json_depth(
            line, depth=depth, in_string=in_string, escape=escape
        )
        if collecting and depth == 0:
            object_text = "".join(buf).strip()
            if object_text.endswith(","):
                object_text = object_text[:-1]
            yield json.loads(object_text)
            collecting = False
            buf = []
        previous_line = line


def load_missing_papers(builder: MainConferenceCorpusBuilder, limit: int) -> list[CorpusPaper]:
    pdf_dir = builder.pdf_root / VENUE.lower() / str(YEAR)
    existing_paper_ids = {
        pdf_path.stem
        for pdf_path in pdf_dir.glob("*.pdf")
    } if pdf_dir.exists() else set()
    papers: list[CorpusPaper] = []
    for record in iter_icml2026_manifest_records(builder.manifest_path):
        if record.get("conference") != VENUE or record.get("year") != YEAR or not record.get("pdf_url"):
            continue
        paper = CorpusPaper.from_manifest_record(record)
        if paper.paper_id in existing_paper_ids:
            continue
        papers.append(paper)
        if limit and len(papers) >= limit:
            break
    return papers


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cdp", default=CDP_BASE)
    parser.add_argument("--limit", type=int, default=0, help="0 = all missing ICML 2026 PDFs")
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--min-interval-seconds", type=float, default=MIN_INTERVAL_FLOOR_SECONDS)
    parser.add_argument("--safety-seconds", type=float, default=30.0)
    parser.add_argument("--challenge-settle-seconds", type=float, default=6.0)
    parser.add_argument("--refresh-retry-seconds", type=float, default=30.0)
    args = parser.parse_args()
    if args.min_interval_seconds < MIN_INTERVAL_FLOOR_SECONDS:
        print(
            f"[{time.strftime('%H:%M:%S')}] raising min interval from "
            f"{args.min_interval_seconds:g}s to {MIN_INTERVAL_FLOOR_SECONDS:g}s for ICML 2026",
            flush=True,
        )
        args.min_interval_seconds = MIN_INTERVAL_FLOOR_SECONDS

    os.environ["OPENREVIEW_COOKIE_FILE"] = str(COOKIE_FILE)
    proxy = settings.proxy.http_proxy
    builder = MainConferenceCorpusBuilder(proxy_url=proxy)

    # Do not let the generic downloader write a subset back as the full manifest.
    builder.save_manifest_checkpoint = lambda *a, **k: None  # type: ignore[method-assign]
    builder.save_manifest = lambda *a, **k: None  # type: ignore[method-assign]

    remaining = load_missing_papers(builder, args.limit)
    print(
        f"[{time.strftime('%H:%M:%S')}] ICML 2026 missing PDFs={len(remaining):,} "
        f"proxy={proxy} cookie_file={COOKIE_FILE}",
        flush=True,
    )

    completed = 0
    while remaining:
        seconds_left = 0.0
        if COOKIE_FILE.exists():
            try:
                seconds_left = clearance_seconds_left(json.loads(COOKIE_FILE.read_text(encoding="utf-8")))
            except Exception:
                seconds_left = 0.0

        usable_count = int((seconds_left - args.safety_seconds) // args.min_interval_seconds)
        if usable_count <= 0:
            if seconds_left > 0:
                sleep_seconds = min(seconds_left + 2, 45)
                print(
                    f"[{time.strftime('%H:%M:%S')}] clearance too close to expiry "
                    f"({seconds_left:.0f}s); sleeping {sleep_seconds:.0f}s before refresh",
                    flush=True,
                )
                time.sleep(max(1.0, sleep_seconds))
            seconds_left = refresh_usable_clearance(
                args.cdp,
                remaining[0].paper_id.removeprefix("icml-2026-"),
                safety_seconds=args.safety_seconds,
                min_interval_seconds=args.min_interval_seconds,
                settle_seconds=args.challenge_settle_seconds,
                retry_seconds=args.refresh_retry_seconds,
            )
            usable_count = int((seconds_left - args.safety_seconds) // args.min_interval_seconds)

        batch_count = min(len(remaining), max(1, args.batch_size), max(1, usable_count))
        batch = remaining[:batch_count]
        print(
            f"[{time.strftime('%H:%M:%S')}] downloading batch size={len(batch)} "
            f"seconds_left={seconds_left:.0f}",
            flush=True,
        )
        import asyncio

        asyncio.run(
            builder.download_papers(
                batch,
                max_concurrent=1,
                min_request_interval_seconds=args.min_interval_seconds,
            )
        )
        completed += len(batch)
        remaining = remaining[batch_count:]
        print(f"[{time.strftime('%H:%M:%S')}] completed={completed:,} remaining={len(remaining):,}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
