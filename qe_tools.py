"""
Small helper layer around Quantum ESPRESSO's pw.x.

Every script in this repo does the same three things:
  1. build a crystal structure (with ASE)
  2. write a pw.x input file and run it
  3. read the results back from QE's XML output

This module keeps steps 2 and 3 in one place so the scripts can focus on the physics.
"""

import json
import os
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Paths and settings
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
PSEUDO_DIR = ROOT / "pseudos"
RUN_DIR = ROOT / "runs"        # raw QE input/output files (not committed to git)
RESULTS_DIR = ROOT / "results"  # plots and summary numbers (committed)

# Number of MPI processes. An M2 has 4 performance cores, so 4 is a good default.
# Override from the terminal with:  export DFT_NPROCS=2
NPROCS = int(os.environ.get("DFT_NPROCS", "4"))

HARTREE_TO_EV = 27.211386245988
RY_TO_EV = HARTREE_TO_EV / 2
BOHR_TO_ANGSTROM = 0.529177210903


# ---------------------------------------------------------------------------
# Writing the input file
# ---------------------------------------------------------------------------

def _format_value(value):
    """Turn a Python value into Fortran namelist syntax."""
    if isinstance(value, bool):
        return ".true." if value else ".false."
    if isinstance(value, str):
        return f"'{value}'"
    return repr(value)


def make_pw_input(atoms, namelists, species, kpoints, labels=None, extra_cards=""):
    """
    Build the text of a pw.x input file.

    atoms       : ASE Atoms object (the crystal)
    namelists   : dict like {"control": {...}, "system": {...}, "electrons": {...}}
    species     : dict {label: (mass, pseudopotential filename)}
    kpoints     : either a tuple (n1, n2, n3) for an automatic Monkhorst-Pack mesh,
                  or an (N, 3) array of k-points in crystal coordinates (band paths)
    labels      : optional list giving each atom's species label. Defaults to the
                  chemical symbol. Needed later when two Eu atoms must be treated as
                  different species (spin up vs spin down in an antiferromagnet).
    extra_cards : extra text appended at the end (for example the HUBBARD card)
    """
    if labels is None:
        labels = atoms.get_chemical_symbols()

    # QE needs to know how many atoms and how many species there are
    system = dict(namelists.get("system", {}))
    system.setdefault("ibrav", 0)
    system["nat"] = len(atoms)
    system["ntyp"] = len(species)
    full = {
        "control": dict(namelists.get("control", {})),
        "system": system,
        "electrons": dict(namelists.get("electrons", {})),
    }
    full["control"].setdefault("pseudo_dir", str(PSEUDO_DIR))
    full["control"].setdefault("outdir", "./tmp")

    lines = []
    for name, params in full.items():
        lines.append(f"&{name}")
        for key, value in params.items():
            lines.append(f"  {key} = {_format_value(value)}")
        lines.append("/")

    lines.append("ATOMIC_SPECIES")
    for label, (mass, pseudo) in species.items():
        lines.append(f"  {label} {mass} {pseudo}")

    lines.append("CELL_PARAMETERS angstrom")
    for vec in atoms.cell:
        lines.append("  " + "  ".join(f"{x:.10f}" for x in vec))

    lines.append("ATOMIC_POSITIONS crystal")
    for label, pos in zip(labels, atoms.get_scaled_positions()):
        lines.append(f"  {label} " + "  ".join(f"{x:.10f}" for x in pos))

    if isinstance(kpoints, tuple):
        lines.append("K_POINTS automatic")
        lines.append("  " + " ".join(str(n) for n in kpoints) + " 0 0 0")
    else:
        kpoints = np.asarray(kpoints)
        lines.append("K_POINTS crystal")
        lines.append(f"  {len(kpoints)}")
        for k in kpoints:
            lines.append("  " + "  ".join(f"{x:.10f}" for x in k) + "  1.0")

    if extra_cards:
        lines.append(extra_cards.strip())

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Running pw.x
# ---------------------------------------------------------------------------

def _mpi_prefix():
    """Command prefix for running in parallel (empty if NPROCS is 1)."""
    if NPROCS <= 1:
        return []
    prefix = ["mpirun", "-np", str(NPROCS)]
    if hasattr(os, "geteuid") and os.geteuid() == 0:   # only matters on Linux servers
        prefix.append("--allow-run-as-root")
    return prefix


def run_pw(name, input_text, workdir):
    """
    Write `name.in` into `workdir`, run pw.x on it, and return the parsed results.

    The output goes to `name.out` in the same folder. If that output already
    exists and finished cleanly, the calculation is skipped and the old result
    is reused, so re-running a script is fast. Delete the folder to force a rerun.
    """
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    infile = workdir / f"{name}.in"
    outfile = workdir / f"{name}.out"

    finished = outfile.exists() and "JOB DONE" in outfile.read_text()
    same_input = infile.exists() and infile.read_text() == input_text
    if not (finished and same_input):
        if shutil.which("pw.x") is None:
            raise RuntimeError("pw.x not found. Did you run 'conda activate eus-dft'?")
        infile.write_text(input_text)
        cmd = _mpi_prefix() + ["pw.x", "-in", infile.name]
        print(f"  running {name} ...", flush=True)
        env = dict(os.environ, OMP_NUM_THREADS="1")
        if sys.platform == "darwin":
            # macOS gives each user a very long temp folder path, which OpenMPI rejects
            env["TMPDIR"] = "/tmp"
        with open(outfile, "w") as out:
            subprocess.run(cmd, cwd=workdir, stdout=out, stderr=subprocess.STDOUT, env=env)
        if "JOB DONE" not in outfile.read_text():
            raise RuntimeError(f"pw.x failed. Look at the end of {outfile}")
    else:
        print(f"  reusing {name}", flush=True)

    prefix = _find_prefix(input_text)
    return read_xml(workdir / "tmp" / f"{prefix}.save" / "data-file-schema.xml")


def _find_prefix(input_text):
    for line in input_text.splitlines():
        if line.strip().startswith("prefix"):
            return line.split("=")[1].strip().strip("'\"")
    return "pwscf"


# ---------------------------------------------------------------------------
# Reading results
# ---------------------------------------------------------------------------

def read_xml(path):
    """
    Pull the useful numbers out of QE's XML output file.

    Returns a dict with energies in eV:
      energy        total energy
      fermi         Fermi level (or highest occupied level for insulators)
      eigenvalues   array (spin, k-point, band)
      occupations   array (spin, k-point, band), 0 to 1
      kpoints       k-points in cartesian units of 2*pi/alat
      magnetization total magnetic moment per cell (Bohr magnetons), 0 if non-magnetic
    """
    root = ET.parse(path).getroot()
    out = root.find("output")
    bs = out.find("band_structure")

    lsda = bs.find("lsda").text.strip() == "true"
    if lsda:
        nup = int(bs.find("nbnd_up").text)
    else:
        nup = int(bs.find("nbnd").text)

    kpts, eigs, occs = [], [], []
    for ks in bs.findall("ks_energies"):
        kpts.append([float(x) for x in ks.find("k_point").text.split()])
        eigs.append(np.array(ks.find("eigenvalues").text.split(), float))
        occs.append(np.array(ks.find("occupations").text.split(), float))
    eigs = np.array(eigs) * HARTREE_TO_EV
    occs = np.array(occs)

    if lsda:   # QE stores spin up and spin down bands one after the other
        eigs = np.stack([eigs[:, :nup], eigs[:, nup:]])
        occs = np.stack([occs[:, :nup], occs[:, nup:]])
    else:
        eigs = eigs[None]
        occs = occs[None]

    fermi_tag = bs.find("fermi_energy")
    if fermi_tag is None:
        fermi_tag = bs.find("highestOccupiedLevel")
    fermi = float(fermi_tag.text) * HARTREE_TO_EV if fermi_tag is not None else None

    mag = out.find("magnetization")
    total_mag = 0.0
    if mag is not None and mag.find("total") is not None:
        total_mag = float(mag.find("total").text)

    return {
        "energy": float(out.find("total_energy/etot").text) * HARTREE_TO_EV,
        "fermi": fermi,
        "eigenvalues": eigs,
        "occupations": occs,
        "kpoints": np.array(kpts),
        "magnetization": total_mag,
    }


def band_edges(result):
    """
    Valence band maximum, conduction band minimum and gap (eV) from a calculation.
    A band counts as occupied if its occupation is above one half.
    """
    e, f = result["eigenvalues"], result["occupations"]
    vbm = e[f > 0.5].max()
    cbm = e[f < 0.5].min()
    return vbm, cbm, max(0.0, cbm - vbm)


def band_edges_by_count(eigenvalues, n_occupied):
    """
    Band edges for a non-spin-polarized insulator, given how many bands are filled.
    Used for band-structure runs, where QE does not report occupations.
    """
    e = eigenvalues[0]
    vbm = e[:, n_occupied - 1].max()
    cbm = e[:, n_occupied].min()
    return vbm, cbm, cbm - vbm


# ---------------------------------------------------------------------------
# Saving summary numbers
# ---------------------------------------------------------------------------

def save_json(name, data):
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / name
    path.write_text(json.dumps(data, indent=2) + "\n")
    return path


def load_json(name):
    path = RESULTS_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run the earlier script first.")
    return json.loads(path.read_text())
