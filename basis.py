import psi4

psi4.set_memory('2 GB')
psi4.core.set_output_file('psi4.out', False)

mol = psi4.geometry("""
0 1
O  0.000000   0.000000   0.000000
H  0.000000   0.757160   0.586260
H  0.000000  -0.757160   0.586260
symmetry c1
""")

psi4.set_options({
    'basis': 'def2-svp',   # <— changed
    'reference': 'rhf',
    'scf_type': 'pk'
})

e_scf, wfn = psi4.energy('scf', return_wfn=True)
psi4.fcidump(wfn, fname='H2O_def2SVP.FCIDUMP')  # spatial-orbital MO integrals
print("Wrote H2O_def2SVP.FCIDUMP")
