import argparse
import json
import os
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BINARY = ROOT / "build" / "SCF"

import re
import math

ITER_RE = re.compile(r"Iter\s+(\d+)")
ENERGY_RE = re.compile(r"Final SCF energy:\s*(-?\d+(?:\.\d+)?)")
NBF_RE = re.compile(r"nbf:\s*(\d+)")
RSS_RE = re.compile(r"peak RSS:\s*(\d+(?:\.\d+)?)\s*MB")
PHASE_RE = re.compile(
    r"^\s*(.*?)\s+(\d+(?:\.\d+)?)\s+(\d+)\s+(\d+(?:\.\d+)?)\s*$"
)

def parse_output(text):
    energy = None
    nbf = None
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
                "calls": int(calls),
                "avg_ms": float(avg_ms),
            }

    return energy, nbf, phases, rss_mb, n_iter


def run_once(xyz, basis, maxiter, econv):
    xyz_path = os.path.join(ROOT, xyz)
    t0 = time.perf_counter()
    proc = subprocess.run(
        [BINARY, xyz_path, basis, str(maxiter), str(econv)],
        capture_output=True, text=True,
    )
    wall = time.perf_counter() - t0
    if proc.returncode != 0:
        raise RuntimeError(f"SCF binary failed:\n{proc.stderr}\n{proc.stdout}")
    energy, nbf, phases, rss_mb, n_iter = parse_output(proc.stdout)
    if energy is None:
        raise RuntimeError(f"Could not parse energy from output:\n{proc.stdout}")
    return wall, energy, nbf, phases, rss_mb, n_iter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", type=str, default=datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    ap.add_argument("--repeats", type=int, default=3,
                    help="runs per system; the fastest wall time is reported")
    ap.add_argument("--tier", nargs="+", default=None, choices=["tiny", "small", "medium", "large"],
                    help="run only these tiers")
    ap.add_argument("--runall", action="store_true",
                    help="run all systems even with run_by_default=false")
    ap.add_argument("--tol", type=float, default=1e-6,
                    help="max |E_cpp - E_psi4| to count as a pass")
    ap.add_argument("--maxiter", type=int, default=200)
    ap.add_argument("--econv", type=float, default=1e-8)
    args = ap.parse_args()

    with open(ROOT / "bench" / "systems.json") as fh:
        systems = json.load(fh)

    ref_path = ROOT / "bench" / "reference.json"
    if not ref_path.exists():
        raise RuntimeError("Psi4 reference not yet exists, regenenerate with `bench/gen_reference.py`")
    with open(ref_path) as fh:
        references = json.load(fh)

    if not BINARY.exists():
        raise RuntimeError(f"SCF binary not found: {BINARY}")
    
    if args.tier:
        systems = {k: v for k, v in systems.items() if v["tier"] in args.tier}
    if not args.runall:
        systems = {k: v for k, v in systems.items() if v["run_by_default"]}
        
    header = (f"{'system':<16} {'nbf':>5} {'|dE|':>10} {'status':>9}"
              f"{'psi4_wall(s)':>16} {'wall(s)':>9} {'ERI(s)':>9} {'JK/it(ms)':>10} {'N_iter':>9}"
              f"{'peakRSS(MB)':>12}")
    print("\n" + header)
    print("-" * len(header))

    all_pass = True
    outputs = {}
    for label, system in systems.items():
        ref = references[label]
        best = None
        peak_rss = float("nan")
        for _ in range(args.repeats):
            wall, energy, nbf, phases, rss_mb, n_iter = run_once(
                system["xyz"], system["basis"], args.maxiter, args.econv)
            if best is None or wall < best[0]:
                best = (wall, energy, nbf, phases, n_iter)
            # Memory is essentially deterministic, so keep the high-water mark.
            if not (peak_rss == peak_rss) or rss_mb > peak_rss:
                peak_rss = rss_mb
            
        wall, energy, nbf, phases, n_iter = best

        if ref is None:
            status, dE = "NOREF", float("nan")
        else:
            ref_wall = ref["wall"]
            dE = abs(energy - ref["energy"])
            ok = dE < args.tol
            all_pass = all_pass and ok
            status = "PASS" if ok else "FAIL"

        eri_s = phases.get("ERI build", {}).get("seconds", float("nan"))
        jk_ms = phases.get("build_JK", {}).get("avg_ms", float("nan"))
        print(f"{label:<16} {nbf:>5} {dE:>10.2e} {status:>9}"
              f"{ref_wall:>16.3f} {wall:>9.3f} {eri_s:>9.3f} {jk_ms:>10.1f} {n_iter:>9}"
              f"{peak_rss:>12.1f}")

        outputs[label] = {
            "nbf": nbf,
            "energy": float(energy),
            "wall(s)": wall,
            "eri(s)": eri_s,
            "jk(ms)": jk_ms
        }

    print("-" * len(header))
    print(f"tolerance = {args.tol:.1e} vs Psi4/PK reference; "
          f"wall = best of {args.repeats} runs")
    
    OUTPUTS = ROOT / "bench" / "outputs"
    OUTPUTS.mkdir(parents=True, exist_ok=True)

    with open(OUTPUTS / f"{args.name}.json", "w") as fh:
        json.dump(outputs, fh, indent=2)
    

if __name__ == "__main__":
    main()
