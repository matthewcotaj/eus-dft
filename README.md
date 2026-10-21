# EuS from first principles

Density functional theory (DFT) calculations of europium sulfide, a Eu²⁺ ferromagnetic semiconductor, using Quantum ESPRESSO.

My undergraduate research was on colloidal EuGdCuS₃ nanorods, where the magnetism comes from the half-filled 4f shell of Eu²⁺. This project asks what standard DFT gets right and wrong about that kind of material. The short answer for f-electron systems is that plain DFT gets it badly wrong, and the project is built around showing that failure and fixing it.

## Roadmap

| Step | What | Status |
|---|---|---|
| 1 | Convergence tests on silicon | Done |
| 2 | Silicon lattice constant and bulk modulus | Done |
| 3 | Silicon band structure | Done |
| 4 | EuS with plain PBE: the 4f states land at the Fermi level | Next |
| 5 | EuS with PBE+U: gap opens, compare gap, lattice constant and moment to experiment | Planned |
| 6 | Exchange constants J₁, J₂ from magnetic configurations, fed into my [Monte Carlo code](https://github.com/matthewcotaj/magnetic-ordering) to predict T_c (experiment 16.5 K) | Stretch goal |

Silicon comes first because it is cheap and well understood, so it validates the workflow before moving to a material where things can go wrong.

## Results so far

### Convergence

Total energy converged to 1 meV/atom at **ecutwfc = 40 Ry** and a **10×10×10 k-mesh**.

![Silicon convergence](results/si_convergence.png)

### Lattice constant

| | PBE (this work) | Experiment |
|---|---|---|
| Lattice constant | 5.470 Å | 5.431 Å |
| Bulk modulus | 88.7 GPa | 98 GPa |

PBE overestimates the lattice constant by 0.7%, the expected direction and size for this functional.

![Silicon equation of state](results/si_lattice.png)

### Band structure

PBE gives an indirect gap of **0.61 eV** against the experimental 1.12 eV, with the conduction band minimum 84% of the way from Γ to X. Underestimating gaps by about half is the well-known band gap problem of semilocal DFT.

![Silicon band structure](results/si_bands.png)

## Setup (macOS, Apple Silicon)

1. Install Miniforge if `conda --version` does not already work in Terminal:

   ```bash
   curl -L -O https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh
   bash Miniforge3-MacOSX-arm64.sh -b
   ~/miniforge3/bin/conda init zsh
   ```

   Then close and reopen Terminal.

2. Create the environment (installs Quantum ESPRESSO, ASE, NumPy and Matplotlib):

   ```bash
   cd eus-dft
   conda env create -f environment.yml
   conda activate eus-dft
   ```

3. Run the steps in order:

   ```bash
   python 01_si_convergence.py
   python 02_si_lattice.py
   python 03_si_bands.py
   ```

Calculations run on 4 cores by default. Change it with `export DFT_NPROCS=2` (or `1` to run without MPI). Raw QE files go to `runs/`. Re-running a script reuses finished calculations, so delete a folder in `runs/` to force a fresh run.

## Repository layout

```
qe_tools.py           writes pw.x inputs, runs them, reads results from QE's XML output
plotting.py           shared figure style
01_si_convergence.py  cutoff and k-point convergence
02_si_lattice.py      equation of state fit
03_si_bands.py        band structure along L-Γ-X-U|K-Γ
pseudos/              pseudopotentials
results/              figures and summary numbers (JSON)
```

## Pseudopotentials

From pslibrary 1.0.0 (PBE, scalar relativistic), generated with QE's `ld1.x`:

- `Si.pbe-n-rrkjus_psl.1.0.0.UPF` (ultrasoft)
- `S.pbe-n-kjpaw_psl.1.0.0.UPF` (PAW)
- `Eu.pbe-spdfn-kjpaw_psl.1.0.0.UPF` (PAW, 4f electrons in the valence)

The Eu potential keeps the 4f shell as valence electrons rather than freezing it into the core. That choice is what lets the calculation describe Eu²⁺ magnetism at all, and also what exposes the failure of plain PBE in step 4.

## References

- P. Giannozzi et al., *J. Phys.: Condens. Matter* **21**, 395502 (2009); **29**, 465901 (2017). Quantum ESPRESSO.
- A. Dal Corso, *Comput. Mater. Sci.* **95**, 337 (2014). pslibrary.
- A. H. Larsen et al., *J. Phys.: Condens. Matter* **29**, 273002 (2017). ASE.
