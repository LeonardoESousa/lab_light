# Validation

2026-10-26 · Python 3.11 · Streamlit 1.58.0 · Windows

Seven scientific and interaction tests passed using `python -m unittest discover -s tests -v` with the source package on PYTHONPATH. The test run exercised both activities, changes to orbital angles and solvent controls, custom molecules, comparison spectra, saved attempts, and the 361-point angle map. Model tests cover spin-adapted basis orthogonality and S², Hamiltonian symmetry, noninteracting limits, energy-zero invariance, localization-dependent couplings, solvent shifts, spectral Jacobians and integrated intensities, and colour robustness.

The Python wheel was built successfully. The automatic-environment launcher's dependency download was not tested in a clean environment because package-index access was unavailable. Existing local scientific dependencies were used for runtime verification. No public hosting deployment was performed.

The local Streamlit server started successfully. Browser approval rejected localhost navigation, so visual browser inspection could not be completed. Automated Streamlit interaction checks reported no app errors.
