"""Launch the gemm_bench executable, parse its timing table + correctness lines,
and rank each GEMM route by GFLOP/s against OpenBLAS. Mirrors benchmark.py's
conventions (subprocess launch, JSON into bench/outputs/)."""

import argparse
import json
import re
import sys
import subprocess
import logging
from datetime import datetime
from pathlib import Path

ROOT    = Path(__file__).resolve().parents[1]
BINARY  = ROOT / "build" / "gemm_bench"
OUT_DIR = ROOT / "bench" / "gemm"
OUT_DIR.mkdir(exist_ok=True)

# [check] <route> <MxNxK> <PASS|FAIL> max_diff=<x>
CHECK_RE = re.compile(
    r"^\[check\]\s+(\S+)\s+(\d+)x(\d+)x(\d+)\s+(PASS|FAIL)\s+max_diff=(\S+)")
# timing::report() row: "<name> <seconds> <calls> <avg_ms>", name carries the shape
PHASE_RE = re.compile(r"^\s*(.*?)\s+(\d+(?:\.\d+)?)\s+(\d+)\s+(\d+(?:\.\d+)?)\s*$")

# 7950X single-core peak: ~4.5 clock(GHz) * 8 doubles/zmm * 2 FMA units * 2 flops/FMA.
PEAK_GFLOPS = 4.5 * 8 * 2 * 2


def parse(text):
    """Returns {shape: {route: {avg_ms, gflops, ok, max_diff}}}."""
    checks = {} # (route, shape) -> (ok, max_diff)
    results = {} # shape -> route -> metrics
    for line in text.splitlines():
        if m := CHECK_RE.match(line):
            route, M, N, K, status, diff = m.groups()
            checks[(route, f"{M}x{N}x{K}")] = (status == "PASS", float(diff))
        elif m := PHASE_RE.match(line):
            name, seconds, calls, avg_ms = m.groups()
            parts = name.strip().split()
            if len(parts) != 2 or "x" not in parts[1]:
                continue # not a "<route> MxNxK" row (e.g. a header slipped through)
            route, shape = parts
            try:
                M, N, K = (int(x) for x in shape.split("x"))
            except ValueError:
                continue
            avg_ms = float(avg_ms)
            gflops = (2.0 * M * N * K) / (avg_ms * 1e6) if avg_ms > 0 else 0.0
            ok, diff = checks.get((route, shape), (None, None))
            results.setdefault(shape, {})[route] = {
                "avg_ms": avg_ms, "gflops": gflops, "ok": ok, "max_diff": diff,
            }
    return results


def report(results, logger):
    for shape in sorted(results, key=lambda s: [int(x) for x in s.split("x")]):
        routes = results[shape]
        ref = routes.get("openblas", {}).get("gflops")
        logger.info(f"\nShape: {shape}")
        logger.info(f"{'route':<13} {'avg_ms':>10} {'%OpenBLAS':>10} "
                     f"{'GFLOP/s':>10} {'%peak':>8} {'check':>6}")
        logger.info("─" * 70)
        # Rank fastest first.
        for route, d in sorted(routes.items(), key=lambda kv: -kv[1]["gflops"]):
            pct_ob = f"{100 * d['gflops'] / ref:.1f}" if ref else "-"
            pct_pk = f"{100 * d['gflops'] / PEAK_GFLOPS:.1f}"
            check = "-" if d["ok"] is None else ("PASS" if d["ok"] else "FAIL")
            logger.info(f"{route:<13} {d['avg_ms']:>10.4f} {pct_ob:>10} "
                         f"{d['gflops']:>10.2f} {pct_pk:>8} {check:>6}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default=datetime.now().strftime("gemm_%Y-%m-%d_%H-%M-%S"))
    ap.add_argument("--sizes", help="comma-separated square dims, e.g. 256,512,1024")
    ap.add_argument("--repeats", type=int)
    args = ap.parse_args()

    if not BINARY.exists():
        raise RuntimeError(f"Binary not found: {BINARY} (build the gemm_bench target)")

    logger = logging.Logger("GEMMBench")
    log_path = OUT_DIR / f"{args.name}.log"

    logger.addHandler(logging.FileHandler(log_path, mode='w'))
    logger.addHandler(logging.StreamHandler(sys.stdout))

    cmd = [str(BINARY)]
    if args.sizes:
        cmd += ["--sizes", args.sizes]
    if args.repeats:
        cmd += ["--repeats", str(args.repeats)]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"gemm_bench failed:\n{proc.stderr}\n{proc.stdout}")

    results = parse(proc.stdout)
    report(results, logger)

    (OUT_DIR / f"{args.name}.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
