# Nanoindentation via Machine Learning Interatomic Potentials (MLIP)
## On-the-Fly (OTF) Active Learning Framework

This repository contains a comprehensive framework for developing robust Moment Tensor Potentials (MTP) for nanoindentation simulations. The core of the project is an **On-the-Fly (OTF) Active Learning** loop that iteratively explores the configuration space using Molecular Dynamics (MD), identifies high-uncertainty configurations, labels them with Density Functional Theory (DFT), and re-trains the potential.

---

## 1. Logic and Mathematical Foundations

### 1.1. Moment Tensor Potentials (MTP)
The MTP represents the energy of an atomic configuration as a sum of local contributions:

$$E(\text{struct}) = \sum_i V(\mathfrak{n}_i)$$

where $V(\mathfrak{n}_i)$ is the potential energy of the local environment $\mathfrak{n}_i$ of atom $i$, expanded in terms of moment tensors.

### 1.2. Uncertainty Quantification: Extrapolation Grade ($\gamma$)
The framework uses a D-optimality criterion to quantify the uncertainty of the MTP for a given configuration. This is expressed as the **extrapolation grade** $\gamma$:

$$\gamma(\mathfrak{n}) = \sqrt{b(\mathfrak{n})^\top \mathbf{A}^{-1} b(\mathfrak{n})}$$

where $b(\mathfrak{n})$ is the basis function vector for environment $\mathfrak{n}$, and $\mathbf{A}$ is the information matrix of the training set.

- **$\gamma \le 1.1$**: The configuration is considered "safe" (interpolation or mild extrapolation).
- **$1.1 < \gamma < 11$**: The configuration is "informative" and selected for DFT labeling.
- **$\gamma \ge 11$**: Extreme extrapolation; the simulation is terminated to prevent non-physical behavior.

### 1.3. Active Learning Workflow
The framework orchestrates a cyclic process:
1.  **Exploration**: Multiple LAMMPS MD runs ($M$ instances) search the phase space.
2.  **Detection**: MLIP monitors $\gamma$ during MD. Configurations exceeding the threshold are saved to `selected.cfg`.
3.  **Selection**: `mlp select-add` filters the most diverse configurations from the pool.
4.  **Labeling**: VASP calculates accurate energies ($E$) and forces ($\mathbf{F}$).
5.  **Training**: The MTP parameters $\theta$ are updated by minimizing:

$$\mathcal{L}(\theta) = \sum \omega_E |E_{DFT} - E_{MTP}|^2 + \sum \omega_F |\mathbf{F}_{DFT} - \mathbf{F}_{MTP}|^2$$

---

## 2. Detailed File Coverage

### 2.1. Orchestration and Control (`sh/`)
-   **`otf_MAIN.sh`**: The main director. Loads user settings from `usr/`, manages cycle directories (`runs/cycle/`), and sequences the calls to sampling, VASP, and training scripts. It includes a global loop controlled by `cycleF` and `termination` (max cycles).
-   **`otf_sampling.sh`**: Executes `mlp_A select-add` with parameters `--weighting=structures --nbh-weight=1 --energy-weight=0`. It identifies "out-of-sample" configurations from `selected.cfg` by comparing them against the existing `trainset.cfg` and `state.mvs`.
-   **`otf_training.sh`**: Automates MTP training using `mlp train`. Key parameters include `--max-iter=1000`, `--stress-weight=0`, and `--auto-min-dist`. It maintains a backup of the current potential (`curr.mtp_bckup`).
-   **`otf_checkfail.sh`**: Scans the `lmpRX/` directories for `selected.cfg`. If found, it increments `sumfails` and appends the configurations to a global queue for selection.
-   **`otf_lmpS2.sh`**: Manages the parallel execution of 128 (or $M$) LAMMPS instances in batches. It monitors `lmpRXfinished` flags to ensure synchronization.
-   **`otf_lmprun.sh`**: Sets `OMP_NUM_THREADS=1` and prepares the run environment for 32 instances. It launches `lmp_serial` with a unique `${seed}` for stochastic diversity.
-   **`otf_genMVS.sh`**: Updates the selection state by running `mlp_A select-add` to generate a new `new_state.mvs`. This captures the "volume" of the updated training set.
-   **`otf_on62.sh`**: Manages remote computation. It uses `scp` to transfer VASP files to server `10.30.16.62`, submits jobs to Slurm, waits for `OUTCAR_copy`, and returns the results.
-   **`otf_vaspEFmulti.sh` / `otf_vaspEFsingle.sh`**: Prepares VASP run directories. In `mode=2`, it applies a critical 25x25x25 (or 15x15x15) lattice modification to the `POSCAR` and applies a `+5.0` translation to atomic coordinates via a `sed`/`bc` loop.
-   **`otf_outcar2cfg.sh`**: Converts VASP `OUTCAR` outputs back to MLIP `.cfg` format using `mlp convert-cfg` with `--elements-order=0`.
-   **`otf_cfg2poscar.sh`**: Converts `.cfg` files to VASP `POSCAR` format for DFT labeling.
-   **`otf_cout.sh`**: A helper script that counts the number of `POSCAR` files in the current cycle and exports the value to `tmp/N`.
-   **`otf_vasprun.sh`**: The Slurm template for remote VASP execution (16 cores, 120-hour limit).
-   **`otf_Mod.sh`**: A standalone utility to process `sampled.cfg` and generate formatted `POSCAR` files with custom lattice parameters and coordinate shifts ($X+ctoff$).

### 2.2. Configuration and Parameters (`mlip/` & `src/`)
-   **`otf_Neigh2.ini`**: The production MLIP config. Sets `select:threshold=1.1`, `select:threshold-break=11`, and loads `state.mvs`. It defines the fitting weights (`energy-weight=1.0`, `force-weight=0.001`, `stress-weight=0.1`).
-   **`otf_ExtSys.ini`**: Specialized config for extended systems. Sets `select:site-en-weight=0.0` and `select:energy-weight=1.0`, focusing on global energy rather than local environments.
-   **`otf_Neigh1.ini`**: Used in Cycle 0. Similar to `Neigh2.ini` but initializes the `new_state.mvs` without an existing state.
-   **`otf_genMTP.ini`**: Used for the initial MTP generation phase.
-   **`otf_NeighDiff1.ini` / `otf_NeighDiff2.ini`**: Used by `sampling.sh` for batch selection (`select:multiple=TRUE`).
-   **`otf_INCAR`**: VASP input parameters: `ENCUT=500`, `EDIFF=1E-6`, `ISMEAR=-1`, `PREC=Accurate`, and `ISYM=0`.
-   **`otf_POTCAR`**: Contains the Projector Augmented Wave (PAW) pseudopotentials for the elements (e.g., Carbon).
-   **`otf_POSCAR`**: Initial geometry file in VASP format.

### 2.3. Structural Data
-   **`otf_9012304.cif`**: Standard CIF file for Diamond (Space Group 227, $Fd\bar{3}m$), used as a reference for the unit cell.
-   **`otf_data.cfgs`**: The initial atomic configurations (e.g., a $5 \times 5 \times 5$ supercell) used to start the simulations.

---

## 3. Directory Structure

| Directory | Purpose |
| :--- | :--- |
| `sh/` | All orchestration and conversion bash scripts (`otf_*.sh`). |
| `usr/` | User-defined parameters (`M`, `MODE`, `DEBUG`, `TERMINATION`). |
| `tmp/` | Temporary state files (`CYCLE`, `SUM`, `N`, `curr.mtp`, `state.mvs`). |
| `src/` | Binaries and templates (`lmp/`, `mlip/`, `vaspEF/`). |
| `runs/` | Generated during execution. Contains data for every cycle. |
| `inp/` | Initial input configurations (`data.cfgs`). |

---

## 4. Usage Instructions

### 4.1. Setup
1.  **Configure Environment**: Ensure `mpirun` and the binaries (`mlp`, `lmp_serial`, `vasp`) are accessible or update the paths in `sh/` scripts.
2.  **Define Settings**: Edit files in `usr/`:
    - `MODE`: 0 (genMTP), 1 (ExtSys), 2 (Neigh).
    - `M`: Total number of parallel LAMMPS instances.
    - `TERMINATION`: Maximum number of training cycles.
    - `DEBUG`: Set to 1 for interactive step-by-step execution.
3.  **Initialize Data**: Place initial `curr.mtp` and `trainset.cfg` in `tmp/`.

### 4.2. Execution
Navigate to the `sh/` directory and execute the main script:
```bash
./MAIN.sh
```

### 4.3. Monitoring
-   Check the `runs/` directory for cycle progress.
-   Monitor `SUM` in `tmp/` to see how many extrapolation events were detected.
-   If `SUM=0`, the loop will terminate, indicating a converged potential.

---
> [!IMPORTANT]
> The remote execution script `otf_on62.sh` assumes SSH key-based authentication with server `10.30.16.62`. Ensure your `id_rsa` is correctly configured.

> [!WARNING]
> High values for `M` or `TERMINATION` can consume significant computational resources (VASP runs especially). Monitor the Slurm queue on the remote server.
