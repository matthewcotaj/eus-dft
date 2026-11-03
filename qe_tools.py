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
HARTREE_PER_BOHR3_TO_KBAR = 294210.2648438959


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


def _run(executable, name, input_text, workdir):
    """
    Write `name.in` into `workdir` and run a QE program on it.

    The output goes to `name.out` in the same folder. If that output already
    exists, finished cleanly and came from the same input, the calculation is
    skipped, so re-running a script is fast. Delete the folder to force a rerun.
    """
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    infile = workdir / f"{name}.in"
    outfile = workdir / f"{name}.out"

    finished = outfile.exists() and "JOB DONE" in outfile.read_text()
    same_input = infile.exists() and infile.read_text() == input_text
    if finished and same_input:
        print(f"  reusing {name}", flush=True)
        return outfile, False

    if shutil.which(executable) is None:
        raise RuntimeError(f"{executable} not found. Did you run 'conda activate eus-dft'?")
    infile.write_text(input_text)
    cmd = _mpi_prefix() + [executable, "-in", infile.name]
    print(f"  running {name} ...", flush=True)
    env = dict(os.environ, OMP_NUM_THREADS="1")
    if sys.platform == "darwin":
        # macOS gives each user a very long temp folder path, which OpenMPI rejects
        env["TMPDIR"] = "/tmp"
    with open(outfile, "w") as out:
        subprocess.run(cmd, cwd=workdir, stdout=out, stderr=subprocess.STDOUT, env=env)
    if "JOB DONE" not in outfile.read_text():
        raise RuntimeError(f"{executable} failed. Look at the end of {outfile}")
    return outfile, True


def run_pw(name, input_text, workdir):
    """
    Run pw.x (see _run) and return the parsed results from its XML output.

    QE writes its results to tmp/<prefix>.save/, and a later nscf or bands run
    with the same prefix overwrites them. So each run's XML is copied to
    <name>.xml right after it finishes, and results are always read from that copy.
    """
    workdir = Path(workdir)
    _, ran = _run("pw.x", name, input_text, workdir)
    saved = workdir / f"{name}.xml"
    if ran or not saved.exists():
        prefix = _find_prefix(input_text)
        shutil.copy(workdir / "tmp" / f"{prefix}.save" / "data-file-schema.xml", saved)
    return read_xml(saved)


def run_projwfc(prefix, workdir, emin=-20.0, emax=15.0, de=0.02):
    """
    Project the wavefunctions of a finished calculation onto atomic orbitals
    (Eu 4f, Eu 5d, S 3p, ...) with projwfc.x. This gives the projected density
    of states, which shows which orbitals make up each band.

    emin and emax are absolute energies in eV, not relative to the Fermi
    level, so pass a window around the Fermi energy of the run.

    Read the results with read_pdos(workdir, prefix).
    """
    text = (
        "&projwfc\n"
        f"  prefix = '{prefix}'\n"
        "  outdir = './tmp'\n"
        f"  filpdos = '{prefix}'\n"
        f"  Emin = {emin}, Emax = {emax}, DeltaE = {de}\n"
        "/\n"
    )
    _run("projwfc.x", f"{prefix}_projwfc", text, workdir)


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
      pressure      pressure in kbar (only if the run had tstress = .true.)
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

    # stress tensor is in Hartree/bohr^3; pressure is minus a third of its trace
    pressure = None
    stress = out.find("stress")
    if stress is not None:
        sigma = np.array(stress.text.split(), float).reshape(3, 3)
        pressure = -np.trace(sigma) / 3 * HARTREE_PER_BOHR3_TO_KBAR

    return {
        "energy": float(out.find("total_energy/etot").text) * HARTREE_TO_EV,
        "fermi": fermi,
        "eigenvalues": eigs,
        "occupations": occs,
        "kpoints": np.array(kpts),
        "magnetization": total_mag,
        "pressure": pressure,
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


def read_pdos(workdir, prefix, species):
    """
    Read the projected DOS files written by projwfc.x.

    species is the same dict passed to make_pw_input; it is used to look up
    the orbital names (5d, 4f, 3p, ...) stored in each pseudopotential file.

    Returns (energies, total, projections), where
      energies     1D array in eV
      total        array (spin, energy) of the total DOS
      projections  dict like {"Eu 4f": array (spin, energy), "S 3p": ...},
                   summed over all atoms of the same species
    Spin index 0 is up, 1 is down. Spin-down values are made negative so they
    plot below the axis, which is the usual convention.
    """
    workdir = Path(workdir)
    tot = np.loadtxt(workdir / f"{prefix}.pdos_tot")
    energies = tot[:, 0]
    sign = np.array([1, -1])[:, None]
    total = tot[:, 1:3].T * sign

    orbital_names = {label: _orbital_names(PSEUDO_DIR / pseudo)
                     for label, (_, pseudo) in species.items()}

    projections = {}
    # file names look like  eus.pdos_atm#1(Eu)_wfc#6(f)
    for path in sorted(workdir.glob(f"{prefix}.pdos_atm#*")):
        atom = path.name.split("(")[1].split(")")[0]
        n_wfc = int(path.name.split("_wfc#")[1].split("(")[0])
        element = atom.rstrip("0123456789")
        label = f"{element} {orbital_names[atom][n_wfc - 1].lower()}"
        data = np.loadtxt(path)[:, 1:3].T * sign     # local DOS, spin up and down
        projections[label] = projections.get(label, 0) + data
    return energies, total, projections


def _orbital_names(upf_path):
    """Orbital labels (like '5D', '4F') of the atomic wavefunctions in a UPF file."""
    root = ET.parse(upf_path).getroot()
    chis = root.find("PP_PSWFC")
    return [chi.get("label") for chi in chis if chi.tag.startswith("PP_CHI")]


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
