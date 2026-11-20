"""
Step 7: exchange constants J1 and J2 of EuS from total energies.

The Eu spins sit on an fcc lattice. Model them with the classical Heisenberg
Hamiltonian

    H = -sum over pairs J_ij S_i.S_j       (each pair once, S = 7/2)

with J1 for the 12 nearest neighbors and J2 for the 6 next-nearest ones.
J > 0 is ferromagnetic. Two antiferromagnetic orderings separate J1 and J2:
AFM-I (alternating (001) planes) and AFM-II (alternating (111) planes). Each is
compared with the ferromagnetic state computed in the same two-Eu cell, so most
numerical errors cancel in the energy difference (see eus.py for the algebra).

The differences are only a few meV per Eu, so these runs use a tighter scf
threshold and a denser k-mesh than the earlier steps.

Experiment (inelastic neutron scattering) gives J1/kB = 0.221 K and
J2/kB = -0.100 K in the convention that counts each pair twice. In the
convention used here that is 0.442 K and -0.200 K.

Outputs:
  results/eus_exchange.json   energies, J1, J2, and a mean-field T_C estimate
"""

from ase.units import kB

import eus
import qe_tools as qe

U_VALUE = 7.0          # eV on Eu 4f, the same value as step 5
KMESH = 8              # primitive-cell equivalent; denser than step 4's 6x6x6
J1_EXP_K = 0.442       # K, each pair counted once
J2_EXP_K = -0.200      # K

settings = qe.load_json("eus_convergence.json")
ecut = settings["ecutwfc_Ry"]
workdir = qe.RUN_DIR / "07_eus_exchange"

energy = {}
for cell in ("I", "II"):
    for afm in (False, True):
        order = "afm" if afm else "fm"
        r = eus.run_magnetic(f"cell{cell}_{order}", workdir, cell, afm,
                             ecut=ecut, kmesh=KMESH, u=U_VALUE)
        energy[cell, order] = r["energy"]
        # a converged AFM state has (close to) zero net moment, FM has 14 muB
        print(f"  cell {cell:<2} {order.upper():<3}  E = {r['energy']:.6f} eV   "
              f"M = {r['magnetization']:6.2f} muB/cell")
        expected = 0.0 if afm else 2 * eus.MOMENT_EXP
        if abs(r["magnetization"] - expected) > 0.5:
            print(f"  WARNING: cell {cell} {order.upper()} did not converge to "
                  "the intended magnetic order. Check the moments in the output "
                  "before using J1, J2.")

j1, j2 = eus.exchange_constants(energy["I", "fm"], energy["I", "afm"],
                                energy["II", "fm"], energy["II", "afm"])
j1_K, j2_K = j1 / kB, j2 / kB

# Mean-field (Curie-Weiss) temperature: kB T = S(S+1)/3 * (12 J1 + 6 J2).
# Mean field ignores fluctuations, so it overestimates T_C; the Monte Carlo
# simulation is what turns J1, J2 into a real T_C prediction.
s = eus.S_EU
t_mf = s * (s + 1) / 3 * (12 * j1_K + 6 * j2_K)
t_mf_exp = s * (s + 1) / 3 * (12 * J1_EXP_K + 6 * J2_EXP_K)

print(f"\nJ1 = {1000 * j1:+.4f} meV = {j1_K:+.3f} K   (experiment {J1_EXP_K:+.3f} K)")
print(f"J2 = {1000 * j2:+.4f} meV = {j2_K:+.3f} K   (experiment {J2_EXP_K:+.3f} K)")
print(f"Mean-field T_C: {t_mf:.1f} K from these J, {t_mf_exp:.1f} K from the "
      f"experimental J (measured T_C {eus.TC_EXP} K)")

qe.save_json("eus_exchange.json", {
    "U_eV": U_VALUE,
    "lattice_constant_A": eus.A_EXP,
    "kmesh_primitive_equivalent": KMESH,
    "convention": "H = -sum over pairs J S_i.S_j, each pair once, S = 7/2",
    "energies_eV": {f"cell_{cell}_{order}": round(e, 8)
                    for (cell, order), e in energy.items()},
    "J1_meV": round(1000 * j1, 5),
    "J2_meV": round(1000 * j2, 5),
    "J1_K": round(j1_K, 4),
    "J2_K": round(j2_K, 4),
    "J1_exp_K": J1_EXP_K,
    "J2_exp_K": J2_EXP_K,
    "Tc_mean_field_K": round(t_mf, 2),
    "Tc_mean_field_from_exp_J_K": round(t_mf_exp, 2),
    "Tc_exp_K": eus.TC_EXP,
})
print("Saved results/eus_exchange.json")
