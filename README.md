# Quasiperiodic Particle Transport Simulations

## Overview

This repository contains the core simulation-generation and analysis workflow used to study the dynamics of mobile particles driven over a fixed quasiperiodic particle structure.

The simulations were generated using **LAMMPS** through the `magcolloids` Python framework developed by Dr. Antonio Ortiz Ambriz ([magcolloids](https://github.com/aortiza/magcolloids)).
The main analysis focuses on:

* Particle velocity and direction of motion
* Angular dependence of particle transport
* Alignment and averaging across independent realizations
* Angular/Fourier analysis
* Mean-squared displacement (MSD)
* Comparison of transport behavior between different system configurations

This repository contains the **base workflow** used throughout the project. Additional parameter-specific and exploratory versions of these scripts were also developed but are not required to understand the main simulation and analysis pipeline.

---

# Physical System

The simulated system consists of two particle populations.

### Type 1 — Fixed particles

Type-1 particles define the quasiperiodic structure.

These particles remain fixed during the production simulations and form the substrate/landscape over which the mobile particles move. The points that define the quasiperiodic structure where developed by Dr. Alan Mendoza 

Base parameters:

* Radius: `1.5 µm`
* Magnetic susceptibility: `0.4`
* Diffusion coefficient: `0 µm²/s`

### Type 2 — Mobile particles

Type-2 particles move through the quasiperiodic structure under the applied magnetic driving field.

Base parameters:

* Radius: `1.3 µm`
* Magnetic susceptibility: `0.4`
* Diffusion coefficient: `0.07 µm²/s`

---

# Simulation Workflow

```text
Quasiperiodic Tiling / Vertices
              │
              ▼
       Remove Duplicates
              │
              ▼
        Crop Geometry
              │
              ▼
        Scale Structure
              │
              ▼
     Fixed Particle Population
              │
              ├─────────────────┐
              │                 │
              │        Generate Random
              │        Mobile Particles
              │                 │
              └────────┬────────┘
                       ▼
               Initial System
                       │
                       ▼
             Strong Damping
                       │
                       ▼
             Reduced Damping
                       │
                       ▼
             FIRE Minimization
                       │
                       ▼
          Relaxed Initial State
                       │
                       ▼
           Production Simulation
                       │
                       ▼
          Magnetic Angle Sweep
                       │
                       ▼
       SLURM + Multiprocessing
                       │
                       ▼
            LAMMPS Trajectories
                       │
                       ▼
                  Analysis
```

---

# Repository Structure

```text
simulation/
│
├── GenerationFile.py
│   Generates the initial particle system.
│   Reads the quasiperiodic vertices, cleans and crops the structure,
│   generates the mobile-particle population, and performs the
│   damping/minimization procedure.
│
├── SimulationFile(uses existing file).py
│   Defines and executes an individual production simulation.
│   Reads a relaxed configuration, reconstructs both particle
│   populations, defines the magnetic field, freezes the substrate,
│   and generates/runs the LAMMPS simulation.
│
├── ClusterParallelFile.py
│   Parallel simulation manager.
│   Uses SLURM array information to distribute configurations and
│   multiprocessing to execute different magnetic-field directions.
│
└── SlurmJob.sub
    SLURM submission script.
    Defines cluster resources, activates the required environment,
    configures LAMMPS, and launches the Python simulation manager.


analysis/
│
├── DataAnalysis1.ipynb
│   Velocity and directional analysis.
│
└── DataAnalysis2.ipynb
    Mean-squared-displacement analysis.
```

---

# Initial Configuration Generation

The initial quasiperiodic structure is provided as a set of vertex coordinates.

The generation script performs the following operations:

1. Load the quasiperiodic vertex coordinates.
2. Remove duplicate vertices.
3. Select the desired spatial region.
4. Scale the quasiperiodic structure.
5. Convert the vertices into the fixed particle population.
6. Generate the mobile particle population at random initial positions.
7. Combine the two populations into the simulation system.

For the `P25` configuration, the number of mobile particles is defined as approximately **25% of the number of fixed quasiperiodic particles**. This can be adapted as needed

---

# Initialization and Relaxation Protocol

Because the mobile particles are initially placed randomly, the generated system is relaxed before the production simulations.

The base initialization protocol is:

```text
Random mobile-particle configuration
              │
              ▼
Strong viscous damping
              │
              ▼
Reduced viscous damping
              │
              ▼
FIRE energy minimization
              │
              ▼
Post-minimization relaxation
              │
              ▼
Production-ready configuration
```

During this procedure, the quasiperiodic particle population remains fixed.

The staged relaxation reduces unfavorable initial particle configurations before the final production dynamics are evaluated.

---

# Production Simulation

The relaxed configuration is loaded and separated into the two particle populations according to the LAMMPS particle-type identifier.

The base magnetic-field parameters are:

| Parameter           |    Value |
| ------------------- | -------: |
| Field magnitude     | `7.2 mT` |
| Frequency           |   `3 Hz` |
| Main field angle    |  `26.9°` |
| Additional tilt     |     `7°` |
| Production timestep | `1e-5 s` |
| Simulation duration |   `15 s` |
| Output frame rate   |  `50 Hz` |

The additional field-tilt direction is varied between independent production simulations.

---

# Angular Sweep

Production simulations are performed for field directions from:

```text
0° → 360°
```

in increments of:

```text
5°
```

This produces **73 angular conditions** for each selected initial configuration.

Each angular condition corresponds to an independent LAMMPS simulation.

---

# Cluster Execution

The simulations were executed using a SLURM-based computing cluster.

The computational workflow contains two levels of task distribution:

### SLURM array

Different quasiperiodic structures/configurations are distributed among SLURM array tasks.

### Python multiprocessing

Within each SLURM task, multiple angular simulations are executed concurrently using Python's `multiprocessing.Pool`.

The base configuration requests:

```text
8 CPUs per SLURM task
20 SLURM array tasks
```

Internal OpenMP, MKL, OpenBLAS, and NumExpr threading is restricted to one thread per worker to prevent CPU oversubscription.

---

# Output Data

Production simulations generate LAMMPS trajectory files containing particle information such as:

```text
Particle ID
Particle Type
x position
y position
z position
simulation timestep/frame
```

The main analysis focuses on **particle type 2**, corresponding to the mobile particle population.

Trajectory files may be compressed as GNU zip:

```text
.lammpstrj.gz
```

due to the large amount of simulation data generated.

---

# Naming Convention

Simulation filenames encode information about the particular system and run.

Example:

```text
MPV2N13_Ne4_P25_R3_100_FR50
```

The base naming convention is:

| Label                 | Meaning                                                                       |
| --------------------- | ----------------------------------------------------------------------------- |
| `N13`                 | Rotational symmetry (13)                                                              |
| `Ne`                  | Version number of substrate                                                              |
| `P25`                 | Mobile-particle population approximately 25% of the fixed-particle population |
| `R`                   | Repeated realization / configuration                                          |
| `FR50`                | Output frame rate of 50 Hz                                                    |
| Final numerical value | Magnetic-field tilt direction in degrees                                      |
| `type 1`              | Fixed quasiperiodic particle population                                       |
| `type 2`              | Mobile particle population                                                    |

Additional naming conventions may appear in parameter-specific versions of the scripts.

---

# Analysis

## Velocity and Directional Analysis

`MultiComparisonV2.ipynb` provides the base workflow for analyzing particle motion.

The general analysis sequence is:

```text
LAMMPS trajectory
       │
       ▼
OVITO trajectory import
       │
       ▼
Particle-level pandas data
       │
       ▼
Select mobile particles
       │
       ▼
Order by particle ID and frame
       │
       ▼
Calculate particle motion
       │
       ▼
Velocity magnitude / direction
       │
       ▼
Angular binning
       │
       ▼
Align independent realizations
       │
       ▼
Count-weighted averaging
       │
       ▼
Angular / Fourier analysis
```

The functions in this notebook were used as the base analysis tools for several different simulation configurations.

---

# Mean-Squared Displacement

`MSDV3.ipynb` contains two MSD implementations.

## Initial-reference MSD

Particle displacement is measured relative to each particle's initial recorded position:

$$
MSD(t)
=
\left\langle
[x(t)-x(0)]^2 +
[y(t)-y(0)]^2
\right\rangle
$$

where the average is taken over the mobile-particle population.

## Time-lag MSD

A second implementation calculates displacement as a function of time lag:

$$
MSD(\tau)
=
\left\langle
[x(t+\tau)-x(t)]^2 +
[y(t+\tau)-y(t)]^2
\right\rangle
$$

using all available particles and valid time origins.

This implementation can then be averaged across repeated realizations and different quasiperiodic structures.

---

# Software Requirements

The core workflow uses:

```text
Python
NumPy
pandas
Matplotlib
OVITO
magcolloids
LAMMPS
SLURM
```

The original cluster environment used the Conda environment:

```text
mag_env
```

Exact package versions were not recorded in the current scripts and may need to be reconstructed depending on the computing environment.

---

# Running the Workflow

A typical workflow is:

### 1. Obtain or generate the quasiperiodic vertices

Generate the quasiperiodic tiling and save its vertex coordinates.

### 2. Generate initial configurations

Run the initialization script to:

```text
clean vertices
→ crop structure
→ create particles
→ relax system
→ minimize system
```

### 3. Store relaxed configurations

Place the resulting initial configurations in the directory expected by the production simulation scripts.

### 4. Configure the computing environment

Set the required paths and environment variables, including:

```bash
BASE_DIR
PYTHONPATH
PATH
```

and ensure that LAMMPS and the required Python environment are available.

### 5. Submit the simulations

Submit the SLURM script.

The SLURM array and Python multiprocessing manager will distribute the individual production simulations.

### 6. Analyze trajectories

Use the analysis notebooks to calculate velocity/directional quantities and MSD from the resulting LAMMPS trajectories.

---

# Important Notes

This repository contains the **core simulation and analysis workflow**, rather than every script created during development.

Many additional versions were produced to test different parameters, system sizes, particle ratios, frame rates, and analysis conditions. Most of these versions use the same base functions documented here.

Some file paths and directory structures are specific to the original Windows workstation and computing-cluster environments and will need to be modified when running the project on another system.

The scripts have intentionally been preserved close to the versions used during the original research rather than being extensively refactored after the project was completed.

---

# Handoff Notes

For future work, the most reusable components are:

* Initial particle generation and relaxation procedure
* Production simulation construction
* Magnetic-field angular sweep
* SLURM/multiprocessing workflow
* Trajectory-to-particle-data conversion
* Angular velocity analysis
* Run alignment and weighted averaging
* Initial-reference MSD
* Time-lag MSD

These functions can be adapted to new quasiperiodic structures, particle concentrations, magnetic-field parameters, or simulation conditions by modifying the relevant input paths and physical parameters.
