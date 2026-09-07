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
BINARY = ROOT / "build" / "ESTZero"
OUT_DIR = ROOT / "bench" / "outputs"
OUT_DIR.mkdir(exist_ok=True)


ITER_RE      = re.compile(r"SCF Iter\s+(\d+)")
ENERGY_RE    = re.compile(r"Final energy:\s*(-?\d+(?:\.\d+)?)")
CCSD_E_RE    = re.compile(r"CCSD energy:\s*(-?\d+(?:\.\d+)?)")
PTRIPLES_RE  = re.compile(r"Perturbative triples correction:\s*(-?\d+(?:\.\d+)?)")
NBF_RE       = re.compile(r"nbf:\s*(\d+)")
RSS_RE       = re.compile(r"peak RSS:\s*(\d+(?:\.\d+)?)\s*MB")
PHASE_RE     = re.compile(r"^\s*(.*?)\s+(\d+(?:\.\d+)?)\s+(\d+)\s+(\d+(?:\.\d+)?)\s*$")
CC_ITER_RE   = re.compile(r"^CC\S*\s+Iteration")
CC_MO_KEY    = "MO Transformation (CC)"
PTRIPLES_KEY = "Perbutative Triples"
MP3_KEY      = "MP3 Energy Calculation"


def _cc_iter_phase(phases):
    """Find whichever '<CC-method> Iteration' phase is present (CCSD today,
    CCSD(T)/CC2/etc. later) without hardcoding the method name."""
    for name, data in phases.items():
        if CC_ITER_RE.match(name):
            return data
    return {}


def parse_output(text):
    energy = nbf = None
    ccsd_e = None
    ptriples = math.nan
    rss_mb = math.nan
    phases = {}
    n_iter = 0
    for line in text.splitlines():
        if m := ENERGY_RE.search(line):
            energy = float(m.group(1))
        elif m := CCSD_E_RE.search(line):
            ccsd_e = float(m.group(1))
        elif m := PTRIPLES_RE.search(line):
            ptriples = float(m.group(1))
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
    return energy, nbf, phases, rss_mb, n_iter, ccsd_e, ptriples


def run_once(xyz, basis, method, reference, mult, charge, maxiter, econv, erischeme, gemmscheme):
    t0   = time.perf_counter()
    proc = subprocess.run(
        [BINARY, os.path.join(ROOT, xyz), basis,
         "--reference", reference, "--method", method,
         "--mult", str(mult), "--charge", str(charge),
         "--max-iter", str(maxiter), "--econv", str(econv), 
         "--eri-scheme", erischeme, "--gemm-scheme", gemmscheme],
        capture_output=True, text=True,
    )
    wall = time.perf_counter() - t0
    if proc.returncode != 0:
        raise RuntimeError(f"SCF binary failed:\n{proc.stderr}\n{proc.stdout}")
    energy, nbf, phases, rss_mb, n_iter, ccsd_e, ptriples = parse_output(proc.stdout)
    if energy is None:
        raise RuntimeError(f"Could not parse energy:\n{proc.stdout}")
    return wall, energy, nbf, phases, rss_mb, n_iter, ccsd_e, ptriples


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


def _print_scf(collected, logger: logging.Logger):
    hdr = (f"{'system':<20} {'ref':>3} {'nbf':>4} {'|dE|':>10} {'ok':>5}"
           f" {'psi4(s)':>8} {'wall(s)':>8} {'ERI(s)':>7} {'JK/it(ms)':>10}"
           f" {'iters':>6} {'RSS(MB)':>8}")
    logger.info(f"\n── SCF {'─' * (len(hdr) - 6)}")
    logger.info(hdr)
    logger.info("─" * len(hdr))
    for label, sys_data in collected.items():
        if "scf" not in sys_data:
            continue
        d = sys_data["scf"]
        logger.info(f"{label:<20} {d['reference']:>3} {d['nbf']:>4} {fmt(d['dE'],'10.2e'):>10} {d['status']:>5}"
              f" {fmt(d['ref_wall'],'8.3f'):>8} {d['wall']:>8.3f} {fmt(d['eri_s'],'7.3f'):>7}"
              f" {fmt(d['jk_ms'],'10.1f'):>10}"
              f" {d['n_iter']:>6} {fmt(d['peak_rss'],'8.1f'):>8}")
    logger.info("─" * len(hdr))

def _json_scf(d):
    return {"wall(s)": d["wall"], "eri(s)": d["eri_s"], "jk(ms)": d["jk_ms"]}


def _print_mp2(collected, logger: logging.Logger):
    hdr = (f"{'system':<20} {'ref':>3} {'nbf':>4} {'|dE|':>10} {'ok':>5}"
           f" {'psi4(s)':>8} {'wall(s)':>8} {'ERI(s)':>7} {'JK/it(ms)':>10}"
           f" {'iters':>6} {'MO(s)':>8}"
           f" {'MP2(ms)':>7} {'RSS(MB)':>8}")
    logger.info(f"\n── MP2 {'─' * (len(hdr) - 6)}")
    logger.info(hdr)
    logger.info("─" * len(hdr))
    for label, sys_data in collected.items():
        if "mp2" not in sys_data:
            continue
        d = sys_data["mp2"]
        logger.info(f"{label:<20} {d['reference']:>3} {d['nbf']:>4} {fmt(d['dE'],'10.2e'):>10} {d['status']:>5}"
              f" {fmt(d['ref_wall'],'8.3f'):>8} {d['wall']:>8.3f} {fmt(d['eri_s'],'7.3f'):>7}"
              f" {fmt(d['jk_ms'],'10.1f'):>10} {d['n_iter']:>6}"
              f" {fmt(d['mo_s'],'7.3f'):>7} {fmt(d['mp2_ms'],'8.1f'):>8}"
              f" {fmt(d['peak_rss'],'8.1f'):>8}")
    logger.info("─" * len(hdr))

def _json_mp2(d):
    return {"wall(s)": d["wall"], "eri(s)": d["eri_s"], "jk(ms)": d["jk_ms"],
            "mo(s)": d["mo_s"], "mp2(ms)": d["mp2_ms"]}


def _print_ccsd(collected, logger: logging.Logger):
    hdr = (f"{'system':<20} {'ref':>3} {'nbf':>4} {'|dE|':>10} {'ok':>5}"
           f" {'psi4(s)':>8} {'wall(s)':>8} {'ERI(s)':>7} {'JK/it(ms)':>10}"
           f" {'scf_it':>6} {'MO(s)':>7} {'CCSD/it(ms)':>12} {'cc_it':>6}"
           f" {'RSS(MB)':>8}")
    logger.info(f"\n── CCSD {'─' * (len(hdr) - 7)}")
    logger.info(hdr)
    logger.info("─" * len(hdr))
    for label, sys_data in collected.items():
        if "ccsd" not in sys_data:
            continue
        d = sys_data["ccsd"]
        logger.info(f"{label:<20} {d['reference']:>3} {d['nbf']:>4} {fmt(d['dE'],'10.2e'):>10} {d['status']:>5}"
              f" {fmt(d['ref_wall'],'8.3f'):>8} {d['wall']:>8.3f} {fmt(d['eri_s'],'7.3f'):>7}"
              f" {fmt(d['jk_ms'],'10.1f'):>10} {d['n_iter']:>6}"
              f" {fmt(d['mo_cc_s'],'7.3f'):>7} {fmt(d['cc_ms'],'12.1f'):>12} {fmt(d['cc_iters'],'6.0f'):>6}"
              f" {fmt(d['peak_rss'],'8.1f'):>8}")
    logger.info("─" * len(hdr))

def _json_ccsd(d):
    return {"wall(s)": d["wall"], "eri(s)": d["eri_s"], "jk(ms)": d["jk_ms"],
            "mo(s)": d["mo_cc_s"], "cc(ms)": d["cc_ms"], "cc_iters": d["cc_iters"]}


def _print_ccsd_t(collected, logger: logging.Logger):
    # One ccsd_t run yields both the CCSD and the (T)-corrected energy; |dE|/ok below track
    # CCSD(T) only (matching every other table). The intermediate CCSD is checked against its
    # own Psi4 reference too, but only ever surfaces as an explicit WARNING line in the run log
    # if it disagrees -- see the main loop.
    hdr = (f"{'system':<20} {'ref':>3} {'nbf':>4} {'|dE|':>10} {'ok':>5}"
           f" {'psi4(s)':>8} {'wall(s)':>8} {'ERI(s)':>7} {'JK/it(ms)':>10}"
           f" {'scf_it':>6} {'MO(s)':>7} {'CCSD/it(ms)':>12} {'cc_it':>6}"
           f" {'CCSD(s)':>8} {'PertT(s)':>8} {'RSS(MB)':>8}")
    logger.info(f"\n── CCSD(T) {'─' * (len(hdr) - 10)}")
    logger.info(hdr)
    logger.info("─" * len(hdr))
    for label, sys_data in collected.items():
        if "ccsd_t" not in sys_data:
            continue
        d = sys_data["ccsd_t"]
        logger.info(f"{label:<20} {d['reference']:>3} {d['nbf']:>4} {fmt(d['dE'],'10.2e'):>10} {d['status']:>5}"
              f" {fmt(d['ref_wall'],'8.3f'):>8} {d['wall']:>8.3f} {fmt(d['eri_s'],'7.3f'):>7}"
              f" {fmt(d['jk_ms'],'10.1f'):>10} {d['n_iter']:>6}"
              f" {fmt(d['mo_cc_s'],'7.3f'):>7} {fmt(d['cc_ms'],'12.1f'):>12} {fmt(d['cc_iters'],'6.0f'):>6}"
              f" {fmt(d['ccsd_only_s'],'8.3f'):>8} {fmt(d['ptriples_s'],'8.3f'):>8} {fmt(d['peak_rss'],'8.1f'):>8}")
    logger.info("─" * len(hdr))

def _json_ccsd_t(d):
    return {"wall(s)": d["wall"], "eri(s)": d["eri_s"], "jk(ms)": d["jk_ms"],
            "mo(s)": d["mo_cc_s"], "cc(ms)": d["cc_ms"], "cc_iters": d["cc_iters"],
            "ccsd(s)": d["ccsd_only_s"], "ptriples(s)": d["ptriples_s"],
            "ptriples(E)": d["ptriples"], "dE_ccsd": d["dE_ccsd"]}


def _print_mp3(collected, logger: logging.Logger):
    hdr = (f"{'system':<20} {'ref':>3} {'nbf':>4} {'|dE|':>10} {'ok':>5}"
           f" {'psi4(s)':>8} {'wall(s)':>8} {'ERI(s)':>7} {'JK/it(ms)':>10}"
           f" {'scf_it':>6} {'MO(s)':>7} {'MP3(s)':>8} {'RSS(MB)':>8}")
    logger.info(f"\n── MP3 {'─' * (len(hdr) - 6)}")
    logger.info(hdr)
    logger.info("─" * len(hdr))
    for label, sys_data in collected.items():
        if "mp3" not in sys_data:
            continue
        d = sys_data["mp3"]
        logger.info(f"{label:<20} {d['reference']:>3} {d['nbf']:>4} {fmt(d['dE'],'10.2e'):>10} {d['status']:>5}"
              f" {fmt(d['ref_wall'],'8.3f'):>8} {d['wall']:>8.3f} {fmt(d['eri_s'],'7.3f'):>7}"
              f" {fmt(d['jk_ms'],'10.1f'):>10} {d['n_iter']:>6}"
              f" {fmt(d['mo_cc_s'],'7.3f'):>7} {fmt(d['mp3_s'],'8.3f'):>8}"
              f" {fmt(d['peak_rss'],'8.1f'):>8}")
    logger.info("─" * len(hdr))

def _json_mp3(d):
    return {"wall(s)": d["wall"], "eri(s)": d["eri_s"], "jk(ms)": d["jk_ms"],
            "mo(s)": d["mo_cc_s"], "mp3(s)": d["mp3_s"]}


METHOD_SPECS = {
    "scf":     {"printer": _print_scf,     "to_json": _json_scf},
    "mp2":     {"printer": _print_mp2,     "to_json": _json_mp2},
    "mp3":     {"printer": _print_mp3,     "to_json": _json_mp3},
    "ccsd":    {"printer": _print_ccsd,    "to_json": _json_ccsd},
    "ccsd_t":  {"printer": _print_ccsd_t,  "to_json": _json_ccsd_t},
}


# Each README.md table tracks ONE fixed system per (method, reference) across versions --
# these are those systems, matched to the labels in bench/systems.json. New versions append a
# row to the existing table rather than replacing it (unlike the GEMM microbenchmark table),
# so the markdown output below prints one row per canonical pair actually present in this run,
# not a full table.
README_CANONICAL = {
    ("scf",    "rhf"): "benzene/def2-svp",
    ("scf",    "uhf"): "ch3/aug-cc-pvtz",
    ("mp2",    "rhf"): "benzene/def2-svp",
    ("mp2",    "uhf"): "ch3/aug-cc-pvtz",
    ("mp3",    "rhf"): "pyrimidine/6-31g",
    ("mp3",    "uhf"): "propargyl/6-31g",
    ("ccsd_t", "rhf"): "pyrimidine/sto-3g",
    ("ccsd_t", "uhf"): "propargyl/sto-3g",
}


def _md_row_scf(d, version):
    return (f"| {version} | {d['wall']:.3f} | {fmt(d['eri_s'],'.3f')} | {fmt(d['jk_ms'],'.1f')} "
            f"| {d['n_iter']} | {fmt(d['peak_rss'],'.1f')} | - |")

def _md_row_mp2(d, version):
    return (f"| {version} | {d['wall']:.3f} | {fmt(d['eri_s'],'.3f')} | {fmt(d['jk_ms'],'.1f')} "
            f"| {d['n_iter']} | {fmt(d['mo_s'],'.3f')} | {fmt(d['mp2_ms'],'.1f')} "
            f"| {fmt(d['peak_rss'],'.1f')} | - |")

def _md_row_mp3(d, version):
    return (f"| {version} | {d['wall']:.3f} | {fmt(d['eri_s'],'.3f')} | {fmt(d['jk_ms'],'.1f')} "
            f"| {d['n_iter']} | {fmt(d['mo_cc_s'],'.3f')} | {fmt(d['mp3_s'],'.3f')} "
            f"| {fmt(d['peak_rss'],'.1f')} | - |")

def _md_row_ccsd_t(d, version):
    return (f"| {version} | {fmt(d['ccsd_only_s'],'.3f')} | {fmt(d['eri_s'],'.3f')} "
            f"| {fmt(d['jk_ms'],'.1f')} | {d['n_iter']} | {fmt(d['mo_cc_s'],'.3f')} "
            f"| {fmt(d['cc_ms'],'.1f')} | {fmt(d['cc_iters'],'.0f')} | {fmt(d['ptriples_s'],'.3f')} "
            f"| {fmt(d['peak_rss'],'.1f')} | - |")

MD_HEADERS = {
    "scf":    ("| version | Total wall (s) | ERI build (s) | JK build (per iter) (ms) | N_SCF_iter | Peak RSS (MB) | Note |",
               "|---|---|---|---|---|---|---|"),
    "mp2":    ("| version | Total wall (s) | ERI build (s) | JK build (per iter) (ms) | N_SCF_iter | MO transform(s) | MP2 energy (ms) | Peak RSS (MB) | Note |",
               "|---|---|---|---|---|---|---|---|---|"),
    "mp3":    ("| version | Total wall (s) | ERI build (s) | JK build (per iter) (ms) | N_SCF_iter | MO transform(s) | MP3 energy (s) | Peak RSS (MB) | Note |",
               "|---|---|---|---|---|---|---|---|---|"),
    "ccsd_t": ("| version | Total CCSD wall (s) | ERI build (s) | JK build (per iter) (ms) | N_SCF_iter | MO transform (s) | CC iteration (ms) | N_CC_iter | PertT(s) | Peak RSS (MB) | Note |",
               "|---|---|---|---|---|---|---|---|---|---|---|"),
}
MD_ROW_FN = {"scf": _md_row_scf, "mp2": _md_row_mp2, "mp3": _md_row_mp3, "ccsd_t": _md_row_ccsd_t}


def report_markdown(collected, version, logger):
    """One row per (method, reference) canonical system actually present in this run, in README.md's column order.
    Created to easily append under the existing table for that section. Notes are empty"""
    logger.info("\n--- README.md rows (append) ---")
    for (method, reference), label in README_CANONICAL.items():
        d = collected.get(label, {}).get(method)
        if d is None:
            continue
        header, sep = MD_HEADERS[method]
        logger.info(f"\n{label} ({method}/{reference}):")
        logger.info(header)
        logger.info(sep)
        logger.info(MD_ROW_FN[method](d, version))


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
    ap.add_argument("--erischeme", choices=["vector", "tensor"], default="tensor",
                    help="ERI storage scheme (default: tensor)")
    ap.add_argument("--gemmscheme", choices=["odometer", "eigen", "blas", "cublas_dgemm", "cublas_sgemm",
                                            "avx_8x8", "avx_16x8", "avx_24x8", "cutensor"], default="blas",
                    help="GEMM scheme for einsum (default: blas)")
    ap.add_argument("--version", default="vX.X.X",
                    help="version string stamped into the README.md row output (e.g. v3.1.2)")
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
        methods = system.get("methods", ["scf"])
        if args.method:
            methods = [m for m in methods if m in args.method]

        sys_ref = references.get(label, {})
        collected[label] = {}
        outputs[label] = {"nbf": sys_ref.get("nbf"), "reference": system.get("reference", "rhf")}

        for method in methods:
            print(f"[mySCF/{method}] {label} ...")

            try:
                best     = None
                peak_rss = math.nan

                for _ in range(args.repeats):
                    wall, energy, nbf, phases, rss_mb, n_iter, ccsd_e, ptriples = run_once(
                        system["xyz"], system["basis"], method,
                        system.get("reference", "rhf"),
                        system.get("mult", 1),
                        system.get("charge", 0),
                        args.maxiter, args.econv, args.erischeme, args.gemmscheme)
                    if best is None or wall < best[0]:
                        best = (wall, energy, nbf, phases, n_iter, ccsd_e, ptriples)
                    if math.isnan(peak_rss) or rss_mb > peak_rss:
                        peak_rss = rss_mb

                wall, energy, nbf, phases, n_iter, ccsd_e, ptriples = best
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

                # ccsd_t gives both the CCSD and (T)-corrected energy from one run. The reported |dE|/status 
                # above tracks CCSD(T) only (matching every other table). The intermediate CCSD is still checked 
                # against its own Psi4 reference here, only appears as an explicit warning when CCSD(T) matching 
                # while CCSD itself doesn't - probably wouldn't happen, but just in case.
                dE_ccsd = math.nan
                if method == "ccsd_t":
                    ccsd_ref = sys_ref.get("ccsd")
                    if ccsd_ref is not None and ccsd_e is not None:
                        dE_ccsd = abs(ccsd_e - ccsd_ref["energy"])
                        if dE_ccsd >= args.tol:
                            logger.info(
                                f"[mySCF/{method}] {label} WARNING: intermediate CCSD |dE|={dE_ccsd:.2e} "
                                f"exceeds tol={args.tol:.1e} vs Psi4 (CCSD(T) total status above is unaffected) "
                                f"-- the CCSD path itself needs checking.")
                            all_pass = False

                eri_s  = phases.get("ERI build",             {}).get("seconds", math.nan)
                mo_s   = phases.get("MO Transformation",     {}).get("seconds", math.nan)
                mp2_ms = phases.get("MP2 Energy Calculation",{}).get("avg_ms",  math.nan)

                mo_cc_s  = phases.get(CC_MO_KEY, {}).get("seconds", math.nan)
                cc_phase = _cc_iter_phase(phases)
                cc_ms    = cc_phase.get("avg_ms", math.nan)
                cc_iters = cc_phase.get("calls",  math.nan)
                ptriples_s = phases.get(PTRIPLES_KEY, {}).get("seconds", math.nan)
                ccsd_only_s = (wall - ptriples_s) if not math.isnan(ptriples_s) else math.nan
                mp3_s = phases.get(MP3_KEY, {}).get("seconds", math.nan)

                jk_phase = phases.get("JK build")
                if jk_phase:
                    jk_ms = jk_phase["avg_ms"]
                else:
                    jka = phases.get("JK build", {}).get("avg_ms", 0.0)
                    jkb = phases.get("JK build", {}).get("avg_ms", 0.0)
                    jk_ms = (jka + jkb) if (jka or jkb) else math.nan

                result = {
                    "nbf": nbf, "wall": wall, "n_iter": n_iter, "peak_rss": peak_rss,
                    "dE": dE, "ref_wall": ref_wall, "status": status,
                    "eri_s": eri_s, "jk_ms": jk_ms, "mo_s": mo_s, "mp2_ms": mp2_ms,
                    "mo_cc_s": mo_cc_s, "cc_ms": cc_ms, "cc_iters": cc_iters,
                    "dE_ccsd": dE_ccsd, "ptriples": ptriples, "ptriples_s": ptriples_s,
                    "ccsd_only_s": ccsd_only_s, "mp3_s": mp3_s,
                    "reference": system.get("reference", "rhf"),
                }
                collected[label][method] = result
                outputs[label][method]   = METHOD_SPECS[method]["to_json"](result)

                print(f"[mySCF/{method}] {label} Done!")

            except Exception as e:
                logger.info(f"[mySCF/{method}] {label} ERROR: {e}")
                all_pass = False
                result = {
                    "nbf": sys_ref.get("nbf"), "wall": math.nan, "n_iter": 0, "peak_rss": math.nan,
                    "dE": math.nan, "ref_wall": math.nan, "status": "ERROR",
                    "eri_s": math.nan, "jk_ms": math.nan, "mo_s": math.nan, "mp2_ms": math.nan,
                    "mo_cc_s": math.nan, "cc_ms": math.nan, "cc_iters": math.nan,
                    "dE_ccsd": math.nan, "ptriples": math.nan, "ptriples_s": math.nan,
                    "ccsd_only_s": math.nan, "mp3_s": math.nan,
                    "reference": system.get("reference", "rhf"),
                }
                collected[label][method] = result
                outputs[label][method]   = {"error": str(e)}

    methods_run = {m for sys_data in collected.values() for m in sys_data}
    for method, spec in METHOD_SPECS.items():
        if method in methods_run:
            spec["printer"](collected, logger)

    logger.info(f"\ntol = {args.tol:.1e} vs Psi4; wall = best of {args.repeats} runs")

    run_params = " ".join(f"{k}={v}" for k, v in vars(args).items() if k != "name")
    logger.info(f"params: {run_params}")

    report_markdown(collected, args.version, logger)

    (OUT_DIR / f"{args.name}.json").write_text(json.dumps(nan_to_null(outputs), indent=2))


if __name__ == "__main__":
    main()
