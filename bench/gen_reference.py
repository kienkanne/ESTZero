import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BINARY = ROOT / "build" / "SCF"

def read_xyz_body(path):
    """Return the atom lines of a standard .xyz file (skip count + comment)."""
    with open(path) as fh:
        lines = fh.read().splitlines()
    return [ln for ln in lines[2:] if ln.strip()]


def main():
    import psi4

    with open(ROOT / "bench" / "systems.json") as fh:
        systems = json.load(fh)

    psi4.set_memory("16 GB")
    psi4.core.be_quiet()
    psi4.set_options({
            "scf_type": "pk",
            "reference": "rhf",
            "e_convergence": 1e-8,
            "d_convergence": 1e-8,
            "maxiter": 200,
            "guess": "core",
        })
    
    # Warm up
    psi4.set_options({"basis": "sto-3g"})
    mol = psi4.geometry("He 0 0 0")
    psi4.energy("scf", molecule=mol)
    psi4.core.clean()

    references = {}
    for label, system in systems.items():
        xyz_path = os.path.join(ROOT, system["xyz"])
        basis = system["basis"]
        psi4.set_options({"basis": basis})

        atom_lines = read_xyz_body(xyz_path)
        geom = "\n".join(atom_lines) + "\n"
        geom += "symmetry c1\nno_reorient\nno_com"

        print(f"[psi4] {label} ...")

        t0 = time.perf_counter()
        mol = psi4.geometry(geom)
        energy, wfn = psi4.energy("scf", molecule=mol, return_wfn=True)
        wall = time.perf_counter() - t0

        references[label] = {
            "nbf": int(wfn.basisset().nbf()),
            "energy": float(energy),
            "wall": wall
        }
        print(f"E = {energy:.8f}  nbf = {wfn.basisset().nbf()}  wall = {wall:.3f}")
        psi4.core.clean()

    out_path = ROOT / "bench" / "reference.json"
    with open(out_path, "w") as fh:
        json.dump(references, fh, indent=2)
    print(f"\nWrote {len(references)} references to {out_path}")


if __name__ == "__main__":
    main()
