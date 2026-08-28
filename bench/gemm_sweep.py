import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gemm_bench as gb

try:
    import matplotlib
    matplotlib.use("Agg")  # headless: no display needed, just write the file
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

OUT_DIR = gb.OUT_DIR

# avx_1x8 is excluded by default; add with --include-avx1x8.
DEFAULT_ROUTES = ["openblas", "eigen", "avx_8x8", "avx_12x8", "avx_16x8", "avx_20x8", "avx_24x8"]

# Chosen to resolve both the broad, gradual degradation seen near 1024 and the sharp,
# single-point spike seen near 2048 (v3.1.5's manual spot-check) on one shared default.
DEFAULT_OFFSETS = [-8, -4, -2, -1, 0, 1, 2, 4, 8]

PALETTE = {
    "openblas": "#111111",
    "eigen": "#666666",
    "avx_8x8": "#1f77b4",
    "avx_12x8": "#2ca02c",
    "avx_16x8": "#d62728",
    "avx_20x8": "#9467bd",
    "avx_24x8": "#ff7f0e",
    "avx_1x8": "#8c564b",
}


def parse_int_list(s):
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def run_sweep(center, offsets, routes, repeats, logger):
    """Returns {shape: {route: {...}}}, same schema as gemm_bench.run_all_perf()."""
    sizes = sorted({center + off for off in offsets if center + off > 0})
    results = {}
    for size in sizes:
        for route in routes:
            stdout, stderr = gb.run_perf_route(route, size, repeats)
            parsed = gb.parse(stdout)
            perf_fields = gb.parse_perf_csv(stderr, logger)
            for shape, route_dict in parsed.items():
                results.setdefault(shape, {}).setdefault(route, {}).update(route_dict[route])
                results[shape][route].update(perf_fields)
    return results


def render_chart(results, center, routes, out_path, metric="gflops", title=None):
    """One line per route, x = matrix size N, y = `metric`; a dashed vertical marker highlights
    the exact power-of-2 `center`."""
    if plt is None:
        raise RuntimeError(
            "matplotlib is required to render the sweep chart. "
            "Run this under `conda activate psi4env` (has it already), "
            "or `pip install matplotlib` in whatever environment you're using."
        )

    sizes = sorted(int(s.split("x")[0]) for s in results)
    if not sizes:
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    for route in routes:
        xs, ys = [], []
        for n in sizes:
            d = results.get(f"{n}x{n}x{n}", {}).get(route)
            if d is None or metric not in d:
                continue
            xs.append(n)
            ys.append(d[metric])
        if not xs:
            continue
        ax.plot(xs, ys, marker="o", markersize=4, linewidth=2,
                color=PALETTE.get(route, "#999999"), label=gb_display_name(route))

    x_min, x_max = min(sizes), max(sizes)
    if x_min <= center <= x_max:
        ax.axvline(center, color="#cc0000", linestyle="--", linewidth=1.5)

    ax.set_xlabel("matrix size N (MxNxK, all equal)")
    ax.set_ylabel({"gflops": "GFLOP/s", "pct_openblas": "% of OpenBLAS"}.get(metric, metric))
    ax.set_xticks(sizes)
    ax.tick_params(axis="x", labelrotation=45)
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.7)
    ax.set_ylim(bottom=0)
    if title:
        ax.set_title(title)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def gb_display_name(route):
    return {"openblas": "OpenBLAS", "eigen": "Eigen"}.get(route, route)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--centers", default="1024,2048",
                     help="comma-separated power-of-2 centers to sweep around")
    ap.add_argument("--offsets", default=",".join(str(o) for o in DEFAULT_OFFSETS),
                     help="comma-separated offsets from each center (can be negative)")
    ap.add_argument("--routes", default=",".join(DEFAULT_ROUTES),
                     help="comma-separated routes to sweep (default excludes avx_1x8)")
    ap.add_argument("--include-avx1x8", action="store_true",
                     help="add avx_1x8 to the default route set")
    ap.add_argument("--repeats", type=int, default=1,
                     help="timed repeats per (route,size) -- default 1, this is a shape-of-the-"
                          "curve sweep, not a precision measurement")
    ap.add_argument("--name", default="sweep", help="output file prefix under bench/outputs_gemm/")
    args = ap.parse_args()

    if not gb.BINARY.exists():
        raise RuntimeError(f"Binary not found: {gb.BINARY} (build the gemm_bench target)")

    centers = parse_int_list(args.centers)
    offsets = parse_int_list(args.offsets)
    routes = args.routes.split(",")
    if args.include_avx1x8 and "avx_1x8" not in routes:
        routes.append("avx_1x8")

    all_routes = gb.discover_routes(None)
    missing = [r for r in routes if r not in all_routes]
    if missing:
        raise RuntimeError(f"no route(s) named {missing}; available: {all_routes}")

    for center in centers:
        logger = logging.Logger("GEMMSweep")
        log_path = OUT_DIR / f"{args.name}_{center}.log"
        logger.addHandler(logging.FileHandler(log_path, mode="w"))
        logger.addHandler(logging.StreamHandler(sys.stdout))

        logger.info(f"=== Sweep around N={center}, offsets={offsets}, "
                    f"routes={routes}, repeats={args.repeats} ===")
        results = run_sweep(center, offsets, routes, args.repeats, logger)

        gb.report(results, logger, noperf=False)
        gb.report_markdown(results, logger, noperf=False)

        import json
        (OUT_DIR / f"{args.name}_{center}.json").write_text(json.dumps(results, indent=2))

        svg_path = OUT_DIR / f"{args.name}_{center}.svg"
        render_chart(results, center, routes, svg_path, metric="gflops",
                     title=f"GEMM throughput near N={center} "
                           f"(1 core, AMD 7950X, single-threaded)")


if __name__ == "__main__":
    main()
