import argparse
import csv
import io
import json
import re
import sys
import subprocess
import logging
from datetime import datetime
from pathlib import Path

ROOT    = Path(__file__).resolve().parents[1]
BINARY  = ROOT / "build" / "gemm_bench"
OUT_DIR = ROOT / "bench" / "outputs_gemm"
OUT_DIR.mkdir(exist_ok=True)

# [check] <route> <MxNxK> <PASS|FAIL> max_diff=<x>
CHECK_RE = re.compile(
    r"^\[check\]\s+(\S+)\s+(\d+)x(\d+)x(\d+)\s+(PASS|FAIL)\s+max_diff=(\S+)")
# timing::report() row: "<name> <seconds> <calls> <avg_ms>", name carries the shape
PHASE_RE = re.compile(r"^\s*(.*?)\s+(\d+(?:\.\d+)?)\s+(\d+)\s+(\d+(?:\.\d+)?)\s*$")

# 7950X single-core peak: ~4.5 clock(GHz) * 8 doubles/zmm * 2 FMA units * 2 flops/FMA.
PEAK_GFLOPS = 4.5 * 8 * 2 * 2

# `perf stat` trackings
PERF_EVENTS = "cycles,instructions,L1-dcache-loads,L1-dcache-load-misses,cache-references,cache-misses"

# Pre-AVX hand-rolled routes, frozen as of v3.1.3 (see CHANGELOG) -- excluded from the default
# route set so routine runs stay avx-focused; still reachable via --legacy or --routes.
LEGACY_ROUTES = {
    "avx_1x8",
    "naive_mnk", "naive_mkn", "mkn_4x1", "mkn_8x1",
    "mnk_4x1", "mnk_1x4", "mnk_8x1", "mnk_1x8", "mnk_4x4", "mnk_8x8",
}


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


def parse_perf_csv(stderr_text, logger):
    """Parse `perf stat -x,` CSV rows into {ipc, l1_miss_pct, llc_miss_pct}.

    Row shape: value,unit,event,time_running_ns,pct_running,metric_value,metric_unit
    perf itself already computes "insn per cycle" and the miss percentages into
    metric_value, so this is structured parsing, not regex-scraping the human table.
    """
    events = {}
    for row in csv.reader(io.StringIO(stderr_text)):
        if len(row) < 6:
            continue  # stray blank line etc.
        value_s, _unit, event, _time_running, pct_running, metric_value = row[:6]
        try:
            value = int(value_s)
        except ValueError:
            continue  # "<not supported>" / "<not counted>" sentinel rows
        events[event.removesuffix(":u")] = {
            "value": value,
            "pct_running": float(pct_running) if pct_running else None,
            "metric_value": float(metric_value) if metric_value else None,
        }

    out = {}
    instr, cycles = events.get("instructions"), events.get("cycles")
    if instr and cycles and cycles["value"]:
        out["ipc"] = instr["value"] / cycles["value"]
    l1 = events.get("L1-dcache-load-misses")
    if l1 and l1["metric_value"] is not None:
        out["l1_miss_pct"] = l1["metric_value"]
    llc = events.get("cache-misses")
    if llc and llc["metric_value"] is not None:
        out["llc_miss_pct"] = llc["metric_value"]

    return out


def discover_routes(only=None):
    """Runs `gemm_bench --list-routes`, returns names in the binary's declared order,
    optionally filtered to `only` (validated -- raises if a requested name doesn't exist)."""
    proc = subprocess.run([str(BINARY), "--list-routes"], capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"--list-routes failed:\n{proc.stderr}")
    all_routes = [r for r in proc.stdout.splitlines() if r.strip()]
    if only is None:
        return all_routes
    missing = [r for r in only if r not in all_routes]
    if missing:
        raise RuntimeError(f"no route(s) named {missing}; available: {all_routes}")
    return only


def run_plain(sizes, repeats, enable_legacy):
    """Existing single-invocation path: binary loops every route internally. Returns stdout."""
    cmd = [str(BINARY)]
    if sizes:
        cmd += ["--sizes", sizes]
    if repeats:
        cmd += ["--repeats", str(repeats)]
    if enable_legacy:
        cmd += ["--legacy"]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"gemm_bench failed:\n{proc.stderr}\n{proc.stdout}")
    return proc.stdout


def run_perf_route(route, size, repeats):
    """One subprocess call: `perf stat -x, -e <events> -- gemm_bench --only <route> ...`.
    stdout carries the [check]/timing lines (parsed with the existing parse()), stderr carries 
    the perf CSV."""
    cmd = ["perf", "stat", "-x,", "-e", PERF_EVENTS, "--",
           str(BINARY), "--only", route, "--sizes", str(size), "--repeats", str(repeats)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"perf stat on route '{route}' failed:\n{proc.stderr}\n{proc.stdout}")
    return proc.stdout, proc.stderr


def run_all_perf(routes, sizes, repeats, logger):
    results = {}
    for size in sizes:
        for route in routes:
            stdout, stderr = run_perf_route(route, size, repeats)
            parsed = parse(stdout)          # exactly one shape, one route inside
            perf_fields = parse_perf_csv(stderr, logger)
            for shape, route_dict in parsed.items():
                results.setdefault(shape, {}).setdefault(route, {}).update(route_dict[route])
                results[shape][route].update(perf_fields)
    return results


def report(results, logger, noperf=False):
    for shape in sorted(results, key=lambda s: [int(x) for x in s.split("x")]):
        routes = results[shape]
        ref = routes.get("openblas", {}).get("gflops")
        logger.info(f"\nShape: {shape}")
        header = (f"{'route':<13} {'avg_ms':>10} {'%OpenBLAS':>10} "
                  f"{'GFLOP/s':>10} {'%peak':>8} {'check':>6}")
        if not noperf:
            header += f" {'IPC':>6} {'L1miss%':>8} {'LLCmiss%':>9}"
        logger.info(header)
        logger.info("─" * (70 if noperf else 96))
        # Rank fastest first.
        for route, d in sorted(routes.items(), key=lambda kv: -kv[1]["gflops"]):
            pct_ob = f"{100 * d['gflops'] / ref:.1f}" if ref else "-"
            pct_pk = f"{100 * d['gflops'] / PEAK_GFLOPS:.1f}"
            check = "-" if d["ok"] is None else ("PASS" if d["ok"] else "FAIL")
            line = (f"{route:<13} {d['avg_ms']:>10.4f} {pct_ob:>10} "
                    f"{d['gflops']:>10.2f} {pct_pk:>8} {check:>6}")
            if not noperf:
                ipc = f"{d['ipc']:.2f}" if "ipc" in d else "-"
                l1  = f"{d['l1_miss_pct']:.2f}" if "l1_miss_pct" in d else "-"
                llc = f"{d['llc_miss_pct']:.2f}" if "llc_miss_pct" in d else "-"
                line += f" {ipc:>6} {l1:>8} {llc:>9}"
            logger.info(line)


def report_markdown(results, logger, noperf=False):
    """Prints the full ranked table in README.md's format. This table gets replaced every patch in
    README.md's GEMM Microbenchmark section. `Note` is left blank."""
    DISPLAY_NAME = {"openblas": "OpenBLAS", "eigen": "Eigen"}  # matches README's existing casing
    for shape in sorted(results, key=lambda s: [int(x) for x in s.split("x")]):
        routes = results[shape]
        ref = routes.get("openblas", {}).get("gflops")
        logger.info(f"\nShape: {shape}")
        if noperf:
            logger.info("| Route | Avg(ms) | %OpenBLAS | GFLOP/s | %Peak | Note |")
            logger.info("| --- | --- | --- | --- | --- | --- |")
        else:
            logger.info("| Route | Avg(ms) | %OpenBLAS | GFLOP/s | %Peak | IPC | L1miss% | LLCmiss% | Note |")
            logger.info("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")   
        for route, d in sorted(routes.items(), key=lambda kv: -kv[1]["gflops"]):
            pct_ob = f"{100 * d['gflops'] / ref:.1f}" if ref else "-"
            pct_pk = f"{100 * d['gflops'] / PEAK_GFLOPS:.1f}"
            name = DISPLAY_NAME.get(route, route)
            if noperf:
                logger.info(f"| {name} | {d['avg_ms']:.3f} | {pct_ob} | {d['gflops']:.2f} | {pct_pk} | - |")
            else:
                logger.info(f"| {name} | {d['avg_ms']:.3f} | {pct_ob} | {d['gflops']:.2f} | {pct_pk} | {d['ipc']:.2f} | {d['l1_miss_pct']:.2f} | {d['llc_miss_pct']:.2f} | - |")
                

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default=datetime.now().strftime("gemm_%Y-%m-%d_%H-%M-%S"))
    ap.add_argument("--sizes", help="comma-separated square dims, e.g. 256,512,1024")
    ap.add_argument("--repeats", type=int)
    ap.add_argument("--legacy", action="store_true",
                     help="include the pre-AVX routes (naive_*/mkn_RxR/mnk_RxR), frozen as of "
                          "v3.1.3 and excluded from the default route set")
    ap.add_argument("--noperf", action="store_true",
                     help="run the binary directly for all paths without perf for separate routes")
    ap.add_argument("--routes", help="comma-separated route names (only valid without --noperf); "
                                      "default: every non-legacy route gemm_bench --list-routes "
                                      "reports (add --legacy to include the frozen pre-AVX ones)")
    args = ap.parse_args()

    if args.routes and args.noperf:
        ap.error("--routes only applies without --noperf ")

    if not BINARY.exists():
        raise RuntimeError(f"Binary not found: {BINARY} (build the gemm_bench target)")

    logger = logging.Logger("GEMMBench")
    log_path = OUT_DIR / f"{args.name}.log"

    logger.addHandler(logging.FileHandler(log_path, mode='w'))
    logger.addHandler(logging.StreamHandler(sys.stdout))

    if not args.noperf:
        if args.routes:
            routes = discover_routes(args.routes.split(","))
        else:
            routes = discover_routes(None)
            if not args.legacy:
                routes = [r for r in routes if r not in LEGACY_ROUTES]
        sizes = [int(s) for s in args.sizes.split(",")] if args.sizes else [1022, 1024, 2045, 2048]
        results = run_all_perf(routes, sizes, args.repeats or 3, logger)
    else:
        stdout = run_plain(args.sizes, args.repeats, args.legacy)
        results = parse(stdout)

    report(results, logger, noperf=args.noperf)
    report_markdown(results, logger, noperf=args.noperf)

    (OUT_DIR / f"{args.name}.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
