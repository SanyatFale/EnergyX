#!/usr/bin/env python3
"""EnergyX data producer — streams IDEAL ticks to the live FastAPI bus.

This script is independent of the Streamlit app.  Run it in a separate
terminal while the dashboard is open in online mode.

Usage
-----
Basic (stream home96, full 14-day window at ~10 ticks/sec):
    python scripts/stream_ideal.py --home home96

Custom window and speed:
    python scripts/stream_ideal.py \\
        --home   home96 \\
        --start  2017-09-01 \\
        --end    2017-09-07 \\
        --every  5          \\  # send every 5th tick (density)
        --speed  0.05       \\  # seconds between ticks (0.05 = 20 ticks/sec)
        --host   http://localhost:8000

Arguments
---------
--home    IDEAL home id (home96 / home128 / home62)
--start   Start date YYYY-MM-DD  (default: home's first date in store)
--end     End date   YYYY-MM-DD  (default: home's last date in store)
--every   Keep every Nth tick    (1 = all, 10 = 10 % density)   [default: 5]
--speed   Seconds to sleep between tick posts                   [default: 0.05]
--host    FastAPI base URL                   [default: http://localhost:8000]
--dry-run Print ticks without posting (for testing)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── repo root on path ─────────────────────────────────────────────────────────
_REPO = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO))


def _post(url: str, body: Dict[str, Any], timeout: float = 10.0) -> Dict[str, Any]:
    data = json.dumps(body).encode()
    req  = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def _check_server(host: str) -> bool:
    try:
        with urllib.request.urlopen(f"{host}/health", timeout=5) as resp:
            d = json.loads(resp.read())
            return d.get("status") == "ok"
    except Exception:
        return False


def _load_ticks(home_id: str, start: Optional[str], end: Optional[str]) -> Any:
    from energyx.data.historic_store import HistoricStore, ParquetBackend
    store_path = _REPO / "data" / "historic_store"
    store = HistoricStore(backend=ParquetBackend(store_path))

    # Default window: use home's configured window from app constants
    _WINDOWS = {
        "home96":  ("2017-09-01", "2017-09-14"),
        "home128": ("2017-10-01", "2017-10-14"),
        "home62":  ("2017-03-01", "2017-03-14"),
    }
    default = _WINDOWS.get(home_id, ("2017-01-01", "2017-01-14"))
    s = datetime.strptime(start or default[0], "%Y-%m-%d")
    e = datetime.strptime(end   or default[1], "%Y-%m-%d")

    print(f"[stream] Loading {home_id} ticks {s.date()} → {e.date()} …", flush=True)
    df = store.read_ticks(home_id, start=s, end=e)
    if df.empty:
        print(f"[stream] ERROR: no data found for {home_id}. Run ingest_ideal.py first.")
        sys.exit(1)

    df = df.sort_values("ts").reset_index(drop=True)
    print(f"[stream] {len(df):,} ticks loaded ({df['sensor_type'].nunique()} sensor types).")
    return df


def _ticks_to_payloads(df: Any, every: int) -> List[Dict[str, Any]]:
    """
    Group rows by timestamp → one TickPayload per unique ts.
    Apply `every` sampling on the sorted timestamp level.
    """
    import pandas as pd

    sampled_ts = sorted(df["ts"].unique())[::every]
    ts_set = set(sampled_ts)
    grouped = df[df["ts"].isin(ts_set)].groupby("ts")

    payloads = []
    for ts, group in grouped:
        ts_str = str(ts) if not hasattr(ts, "isoformat") else ts.isoformat().replace("T", " ")[:19]
        readings = []
        for _, row in group.iterrows():
            readings.append({
                "sensor_id":   str(row.get("sensor_id", "unknown")),
                "sensor_type": str(row.get("sensor_type", "unknown")),
                "value":       float(row.get("value", 0.0)),
                "unit":        str(row.get("unit", "watts")),
            })
        payloads.append({
            "home_id":  group["home_id"].iloc[0] if "home_id" in group.columns else "unknown",
            "ts":       ts_str,
            "readings": readings,
        })

    payloads.sort(key=lambda p: p["ts"])
    return payloads


def main() -> None:
    parser = argparse.ArgumentParser(description="EnergyX IDEAL data streamer")
    parser.add_argument("--home",    default="home96")
    parser.add_argument("--start",   default=None,                help="YYYY-MM-DD")
    parser.add_argument("--end",     default=None,                help="YYYY-MM-DD")
    parser.add_argument("--every",   type=int,   default=5,       help="Keep every Nth tick")
    parser.add_argument("--speed",   type=float, default=0.05,    help="Seconds between posts")
    parser.add_argument("--host",    default="http://localhost:8000")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    host = args.host.rstrip("/")

    # ── 1. Check server ───────────────────────────────────────────────────────
    if not args.dry_run:
        if not _check_server(host):
            print(f"[stream] ERROR: cannot reach {host}/health")
            print(f"         Start the server first:  uvicorn energyx.api.main:app --port 8000")
            sys.exit(1)
        print(f"[stream] Server OK at {host}")

    # ── 2. Load data ──────────────────────────────────────────────────────────
    df = _load_ticks(args.home, args.start, args.end)
    payloads = _ticks_to_payloads(df, every=args.every)
    print(f"[stream] {len(payloads):,} tick payloads after every={args.every} sampling.")
    rate_str = f"{1/args.speed:.0f} ticks/sec" if args.speed > 0 else "unlimited"
    print(f"[stream] Estimated duration: {len(payloads) * args.speed:.1f}s "
          f"at {args.speed}s/tick ({rate_str})")

    if args.dry_run:
        print("[stream] DRY RUN — first 3 payloads:")
        for p in payloads[:3]:
            print(json.dumps(p, indent=2))
        return

    # ── 3. Start session ──────────────────────────────────────────────────────
    try:
        resp = _post(f"{host}/stream/start", {"home_id": args.home})
        print(f"[stream] Session started: {resp}")
    except Exception as e:
        print(f"[stream] ERROR starting session: {e}")
        sys.exit(1)

    # ── 4. Stream ticks ───────────────────────────────────────────────────────
    n_sent = 0
    n_err  = 0
    t0     = time.perf_counter()

    print(f"[stream] Streaming {len(payloads):,} ticks … (Ctrl-C to stop early)")
    for i, payload in enumerate(payloads):
        try:
            _post(f"{host}/ingest/tick", payload, timeout=5.0)
            n_sent += 1
        except Exception as e:
            n_err += 1
            if n_err <= 5:
                print(f"[stream] WARN tick {i}: {e}", flush=True)

        if args.speed > 0:
            time.sleep(args.speed)

        if (i + 1) % 100 == 0:
            elapsed = time.perf_counter() - t0
            rate    = n_sent / elapsed if elapsed > 0 else 0
            print(f"[stream]  {n_sent:,}/{len(payloads):,} ticks sent "
                  f"({rate:.1f}/s, {n_err} errors)", flush=True)

    # ── 5. Signal completion ──────────────────────────────────────────────────
    elapsed = time.perf_counter() - t0
    try:
        resp = _post(f"{host}/stream/complete", {})
        print(f"\n[stream] ✓ Complete. {n_sent:,} ticks in {elapsed:.1f}s "
              f"({n_sent/elapsed:.1f}/s). Errors: {n_err}")
        print(f"[stream] Server status: {resp}")
    except Exception as e:
        print(f"[stream] ERROR signalling completion: {e}")

    print("[stream] Done. Switch the dashboard to Offline mode to analyse collected data.")


if __name__ == "__main__":
    main()
