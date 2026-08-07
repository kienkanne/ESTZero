import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_xyz_body(path):
    with open(path) as fh:
        lines = fh.read().splitlines()
    return [ln for ln in lines[2:] if ln.strip()]


def main():
    import psi4

    with open(ROOT / "bench" / "systems.json") as fh:
        systems = json.load(fh)

    psi4.set_memory("16 GB")
    psi4.core.be_quiet()

    ref_path = ROOT / "bench" / "reference.json"
    references = {}

    for label, system in systems.items():
        xyz_path  = os.path.join(ROOT, system["xyz"])
        basis     = system["basis"]
        methods   = system.get("methods", ["scf"])
        reference = system.get("reference", "rhf")
        mult      = system.get("mult", 1)
        charge    = system.get("charge", 0)

        atom_lines = read_xyz_body(xyz_path)
        geom = f"{charge} {mult}\n" + "\n".join(atom_lines) + "\nsymmetry c1\nno_reorient\nno_com"

        entry = references.setdefault(label, {})

        for method in methods:
            print(f"[psi4/{reference}/{method}] {label} ...")

            psi4.set_options({
                "basis":         basis,
                "scf_type":      "pk",
                "reference":     reference,
                "e_convergence": 1e-8,
                "d_convergence": 1e-8,
                "maxiter":       200,
                "guess":         "core",
            })
            if method == "mp2":
                psi4.set_options({"mp2_type": "conv", "freeze_core": False})

            mol = psi4.geometry(geom)
            t0 = time.perf_counter()
            psi4_method = "hf" if method == "scf" else method
            energy, wfn = psi4.energy(psi4_method, molecule=mol, return_wfn=True)
            wall = time.perf_counter() - t0

            entry["nbf"] = int(wfn.basisset().nbf())
            entry[method] = {"energy": float(energy), "wall": wall}
            print(f"  E = {energy:.10f}  nbf = {entry['nbf']}  wall = {wall:.3f}s")
            psi4.core.clean()

    ref_path.write_text(json.dumps(references, indent=2))
    print(f"\nWrote {len(references)} systems to {ref_path}")


if __name__ == "__main__":
    main()
