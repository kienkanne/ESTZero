import argparse
import json
import math
import os
import re
import subprocess
import time
import logging
import sys
from datetime import datetime
from pathlib import Path


ROOT   = Path(__file__).resolve().parents[1]
BINARY = ROOT / "build" / "SCF"
OUT_DIR = ROOT / "bench" / "outputs"
OUT_DIR.mkdir(exist_ok=True)


ITER_RE   = re.compile(r"Iter\s+(\d+)")
ENERGY_RE = re.compile(r"Final SCF energy:\s*(-?\d+(?:\.\d+)?)")
NBF_RE    = re.compile(r"nbf:\s*(\d+)")
RSS_RE    = re.compile(r"peak RSS:\s*(\d+(?:\.\d+)?)\s*MB")
PHASE_RE  = re.compile(r"^\s*(.*?)\s+(\d+(?:\.\d+)?)\s+(\d+)\s+(\d+(?:\.\d+)?)\s*$")



def parse_output(text):
    energy = nbf = None
    rss_mb = math.nan
    phases = {}
    n_iter = 0
    for line in text.splitlines():
        if m := ENERGY_RE.search(line):
            energy = float(m.group(1))
        elif m := NBF_RE.search(line):
            nbf = int(m.group(1))
        elif m := RSS_RE.search(line):
            rss_mb = float(m.group(1))
        elif m := ITER_RE.search(line):
            n_iter = int(m.group(1))
        elif m := PHASE_RE.match(line):
            name, seconds, calls, avg_ms = m.groups()
            phases[name.strip()] = {
                "seconds": float(seconds),
                "calls":   int(calls),
                "avg_ms":  float(avg_ms),
            }
    return energy, nbf, phases, rss_mb, n_iter


def run_once(xyz, basis, method, maxiter, econv):
    t0   = time.perf_counter()
    proc = subprocess.run(
        [BINARY, os.path.join(ROOT, xyz), basis,
         "--method", method, "--max-iter", str(maxiter), "--tolerance", str(econv)],
        capture_output=True, text=True,
    )
    wall = time.perf_counter() - t0
    if proc.returncode != 0:
        raise RuntimeError(f"SCF binary failed:\n{proc.stderr}\n{proc.stdout}")
    energy, nbf, phases, rss_mb, n_iter = parse_output(proc.stdout)
    if energy is None:
        raise RuntimeError(f"Could not parse energy:\n{proc.stdout}")
    return wall, energy, nbf, phases, rss_mb, n_iter


def fmt(val, spec, missing="-"):
    return missing if (isinstance(val, float) and math.isnan(val)) else format(val, spec)


def nan_to_null(obj):
    """Recursively replace float NaN with None so json.dumps produces valid JSON."""
    if isinstance(obj, float) and math.isnan(obj):
        return None
    if isinstance(obj, dict):
        return {k: nan_to_null(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [nan_to_null(v) for v in obj]
    return obj


def _print_rhf(collected, logger: logging.Logger):
    hdr = (f"{'system':<20} {'nbf':>4} {'|dE|':>10} {'ok':>5}"
           f" {'psi4(s)':>8} {'wall(s)':>8} {'ERI(s)':>7} {'JK/it(ms)':>10}"
           f" {'iters':>6} {'RSS(MB)':>8}")
    logger.info(f"\n── RHF {'─' * (len(hdr) - 6)}")
    logger.info(hdr)
    logger.info("─" * len(hdr))
    for label, sys_data in collected.items():
        if "rhf" not in sys_data:
            continue
        d = sys_data["rhf"]
        logger.info(f"{label:<20} {d['nbf']:>4} {fmt(d['dE'],'10.2e'):>10} {d['status']:>5}"
              f" {fmt(d['ref_wall'],'8.3f'):>8} {d['wall']:>8.3f} {fmt(d['eri_s'],'7.3f'):>7}"
              f" {fmt(d['jk_ms'],'10.1f'):>10}"
              f" {d['n_iter']:>6} {fmt(d['peak_rss'],'8.1f'):>8}")
    logger.info("─" * len(hdr))

def _json_rhf(d):
    return {"wall(s)": d["wall"], "eri(s)": d["eri_s"], "jk(ms)": d["jk_ms"]}


def _print_mp2(collected, logger: logging.Logger):
    hdr = (f"{'system':<20} {'nbf':>4} {'|dE|':>10} {'ok':>5}"
           f" {'psi4(s)':>8} {'wall(s)':>8} {'ERI(s)':>7} {'JK/it(ms)':>10}"
           f" {'MO(s)':>7} {'MP2(ms)':>8}"
           f" {'iters':>6} {'RSS(MB)':>8}")
    logger.info(f"\n── MP2 {'─' * (len(hdr) - 6)}")
    logger.info(hdr)
    logger.info("─" * len(hdr))
    for label, sys_data in collected.items():
        if "mp2" not in sys_data:
            continue
        d = sys_data["mp2"]
        logger.info(f"{label:<20} {d['nbf']:>4} {fmt(d['dE'],'10.2e'):>10} {d['status']:>5}"
              f" {fmt(d['ref_wall'],'8.3f'):>8} {d['wall']:>8.3f} {fmt(d['eri_s'],'7.3f'):>7}"
              f" {fmt(d['jk_ms'],'10.1f'):>10}"
              f" {fmt(d['mo_s'],'7.3f'):>7} {fmt(d['mp2_ms'],'8.1f'):>8}"
              f" {d['n_iter']:>6} {fmt(d['peak_rss'],'8.1f'):>8}")
    logger.info("─" * len(hdr))

def _json_mp2(d):
    return {"wall(s)": d["wall"], "eri(s)": d["eri_s"], "jk(ms)": d["jk_ms"],
            "mo(s)": d["mo_s"], "mp2(ms)": d["mp2_ms"]}


METHOD_SPECS = {
    "rhf": {"printer": _print_rhf, "to_json": _json_rhf},
    "mp2": {"printer": _print_mp2, "to_json": _json_mp2},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default=datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    ap.add_argument("--repeats", type=int, default=3,
                    help="runs per (system, method); fastest wall time is reported")
    ap.add_argument("--tier", nargs="+", choices=["tiny", "small", "medium", "large", "massive"])
    ap.add_argument("--method", nargs="+", choices=list(METHOD_SPECS),
                    help="limit to these methods (default: all listed per system)")
    ap.add_argument("--runall", action="store_true",
                    help="include systems with run_by_default=false")
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--maxiter", type=int, default=200)
    ap.add_argument("--econv", type=float, default=1e-8)
    args = ap.parse_args()

    with open(ROOT / "bench" / "systems.json") as fh:
        systems = json.load(fh)
    ref_path = ROOT / "bench" / "reference.json"
    if not ref_path.exists():
        raise RuntimeError("Missing bench/reference.json. Run bench/gen_reference.py first.")
    with open(ref_path) as fh:
        references = json.load(fh)
    if not BINARY.exists():
        raise RuntimeError(f"Binary not found: {BINARY}")

    logger = logging.Logger("mySCF")

    log_path = OUT_DIR / f"{args.name}.log"

    logger.addHandler(logging.FileHandler(log_path, mode='w'))
    logger.addHandler(logging.StreamHandler(sys.stdout))

    if args.tier:
        systems = {k: v for k, v in systems.items() if v["tier"] in args.tier}
    if not args.runall:
        systems = {k: v for k, v in systems.items() if v.get("run_by_default", True)}

    all_pass  = True
    collected = {}  # label -> {method -> rich result dict}  (for printing)
    outputs   = {}  # label -> {nbf, method -> slim metrics}  (for JSON)

    for label, system in systems.items():
        methods = system.get("methods", ["rhf"])
        if args.method:
            methods = [m for m in methods if m in args.method]

        sys_ref = references.get(label, {})
        collected[label] = {}
        outputs[label]   = {"nbf": sys_ref.get("nbf")}

        for method in methods:
            print(f"[mySCF/{method}] {label} ...")

            best     = None
            peak_rss = math.nan

            for _ in range(args.repeats):
                wall, energy, nbf, phases, rss_mb, n_iter = run_once(
                    system["xyz"], system["basis"], method, args.maxiter, args.econv)
                if best is None or wall < best[0]:
                    best = (wall, energy, nbf, phases, n_iter)
                if math.isnan(peak_rss) or rss_mb > peak_rss:
                    peak_rss = rss_mb

            wall, energy, nbf, phases, n_iter = best
            outputs[label].setdefault("nbf", nbf)

            mref = sys_ref.get(method)
            if mref is None:
                status, dE, ref_wall = "NOREF", math.nan, math.nan
            else:
                dE       = abs(energy - mref["energy"])
                ref_wall = mref.get("wall", math.nan)
                ok       = dE < args.tol
                all_pass = all_pass and ok
                status   = "PASS" if ok else "FAIL"

            eri_s  = phases.get("ERI build",             {}).get("seconds", math.nan)
            jk_ms  = phases.get("build_JK",              {}).get("avg_ms",  math.nan)
            mo_s   = phases.get("MO Transformation",     {}).get("seconds", math.nan)
            mp2_ms = phases.get("MP2 Energy Calculation",{}).get("avg_ms",  math.nan)

            result = {
                "nbf": nbf, "wall": wall, "n_iter": n_iter, "peak_rss": peak_rss,
                "dE": dE, "ref_wall": ref_wall, "status": status,
                "eri_s": eri_s, "jk_ms": jk_ms, "mo_s": mo_s, "mp2_ms": mp2_ms,
            }
            collected[label][method] = result
            outputs[label][method]   = METHOD_SPECS[method]["to_json"](result)

            print(f"[mySCF/{method}] {label} Done!")

    methods_run = {m for sys_data in collected.values() for m in sys_data}
    for method, spec in METHOD_SPECS.items():
        if method in methods_run:
            spec["printer"](collected, logger)

    logger.info(f"\ntol = {args.tol:.1e} vs Psi4; wall = best of {args.repeats} runs")

    (OUT_DIR / f"{args.name}.json").write_text(json.dumps(nan_to_null(outputs), indent=2))


if __name__ == "__main__":
    main()
