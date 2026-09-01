#This code uses a premade structure, it is very very very important to check before if the premade file is stable, it can be in ovito
#or checking the log files in the generation code


import os
from pathlib import Path
import re
import pandas as pd
import numpy as np
import subprocess

# magcolloids provides the particle, magnetic-field, world, unit, and
# simulation objects used to construct the colloidal-particle simulation.
import magcolloids as mgc
import gc


# Unit registry provided by magcolloids. This allows physical quantities
# such as micrometers, seconds, millitesla, and degrees to carry explicit units.
ureg = mgc.ureg

# pandas IndexSlice helper. It is defined here for convenient indexed slicing
# if required by this workflow.
idx = pd.IndexSlice


# Define the main directory in which simulation folders and outputs are stored.
#
# If the environment variable BASE_DIR has been defined, that location is used.
# Otherwise, the default is:
#     ~/work/Vertices_Teselas
#
# Path(...).resolve() converts the resulting location into an absolute path.
BASE_DIR = Path(os.environ.get("BASE_DIR", Path.home() / "work" / "Vertices_Teselas")).resolve()

# Create the output directory and any missing parent directories.
# exist_ok=True prevents an error when the directory already exists.
BASE_DIR.mkdir(parents=True, exist_ok=True)


def Readtrj(trjfile):
    """
    Read particle information from a LAMMPS trajectory file.

    The relevant trajectory section begins after the line 'ITEM: ATOMS'.
    Each particle record is interpreted as:

        particle ID, particle type, x position, y position, z position

    The collected records are returned as a NumPy array, where each row
    corresponds to one particle.
    """

    # Empty Python list used to accumulate one particle record at a time.
    coord = []

    # Open the trajectory file and load its text lines into memory.
    with open(trjfile) as f:
        lines = f.readlines()

    # Locate the beginning of the particle-data section.
    #
    # 'start' stores the line index immediately after "ITEM: ATOMS",
    # which corresponds to the first particle record.
    start = None
    for i, line in enumerate(lines):
        if line.startswith("ITEM: ATOMS"):
            start = i + 1  

    # Process the trajectory lines belonging to the particle-data section.
    for line in lines[start:]:

        # A new "ITEM:" header marks the end of the current particle block.
        if line.startswith("ITEM:"): 
            break

        # Split the whitespace-separated LAMMPS record into individual values.
        parts = line.split()

        # Particle identifiers and particle types are stored as integers.
        atomid = int(parts[0])
        atype = int(parts[1])

        # Particle coordinates are stored as floating-point values.
        # The three values represent the x, y, and z positions.
        x, y, z = map(float, parts[2:5])

        # Store the complete particle record as:
        # [particle ID, particle type, x, y, z]
        coord.append([atomid, atype, x, y, z])

    # Convert the Python list of particle records into a NumPy array.
    # This representation allows the particle populations and coordinate
    # columns to be filtered efficiently later in the script.
    return np.array(coord)


def lammps_finished_ok(run_dir, file_name):
    """
    Determine whether a particular LAMMPS simulation has already completed.

    This prevents simulations that have successfully finished from being
    executed again when a batch of runs is restarted.

    A run is considered complete when:
      1. Its LAMMPS log file exists.
      2. The log file is not empty.
      3. The end of the log contains the LAMMPS completion message
         "Total wall time".
    """

    # Construct the expected path of the LAMMPS log file for this run.
    log_file = run_dir / f"{file_name}.log"

    # A missing log file means that no completed simulation can be confirmed.
    if not log_file.exists():
        return False

    # Obtain the log-file size in bytes.
    size = log_file.stat().st_size

    # An empty log file is also treated as an incomplete run.
    if  size == 0:
        return False
  
    # Open the log as binary data and inspect only its final portion.
    #
    # Reading the last 20,000 bytes avoids loading an entire potentially
    # large simulation log simply to check its completion status.
    with open(log_file, "rb") as f:
        n = 20000

        # If the log is smaller than 20,000 bytes, reading begins at byte 0.
        offset = max(0, size-n)

        # Move the file pointer close to the end of the log.
        f.seek(offset,0)

        # Decode the selected section as text.
        # Undecodable characters are ignored rather than causing an exception.
        tail = f.read().decode(errors="ignore")
    
    # LAMMPS prints "Total wall time" at the end of a successfully completed run.
    # The function therefore returns a Boolean value:
    # True  -> completed simulation
    # False -> completion could not be confirmed
    return "Total wall time" in tail
    

def redefine_field(field,tilt_angle,tilt_dir):
    """
    Modify the components of the externally applied magnetic field.

    The original magcolloids field object supplies the field magnitude,
    frequency, and main angular parameter. This function introduces an
    additional tilt defined by:

        tilt_angle : magnitude of the tilt
        tilt_dir   : direction/orientation of that tilt

    The resulting x, y, and z components are written as LAMMPS-compatible
    expressions that vary with simulation time.
    """

    # Calculate the x-oriented projection produced by the field tilt.
    #
    # tilt_angle and tilt_dir are unit-aware quantities. Their magnitudes
    # are extracted, converted from degrees to radians, and used in the
    # trigonometric calculation.
    sinTAcosTD = np.sin(np.deg2rad(tilt_angle.magnitude))*np.cos(np.deg2rad(tilt_dir.magnitude))

    # Calculate the corresponding y-oriented projection of the tilt.
    sinTAsinTD = np.sin(np.deg2rad(tilt_angle.magnitude))*np.sin(np.deg2rad(tilt_dir.magnitude))

    # Redefine the time-dependent x component of the magnetic field.
    #
    # The expression is stored as a string because it will later become
    # part of the generated LAMMPS simulation input.
    field.fieldx = f'v_Bmag*(sin(v_freq*time*2*PI)*sin(v_theta)+cos(v_theta)*{sinTAcosTD})'

    # Redefine the time-dependent y component.
    field.fieldy = f'v_Bmag*(cos(v_freq*time*2*PI)*sin(v_theta)+cos(v_theta)*{sinTAsinTD})'

    # Define the z component using the projection of the imposed tilt
    # along the vertical direction.
    field.fieldz = f'v_Bmag*(cos(v_theta)*{np.cos(np.deg2rad(tilt_angle.magnitude))})'

    # Return the same field object with its three components redefined.
    return field


def RegularRun(lmpin: Path,rad, parti1):
    """
    Modify a generated LAMMPS input file before the simulation is executed.

    In this system, the first particle population represents the particles
    that must remain fixed during the driven simulation. Their number is
    obtained from parti1.

    The corresponding particle IDs are assigned to a LAMMPS group called
    'frozen', while all remaining particles form the 'mobile' group.
    A zero-force constraint is then applied to the frozen population.

    The resulting commands are inserted immediately before the first
    LAMMPS 'run' command.
    """


    # Read the complete generated LAMMPS input file as a Python string.
    txt = lmpin.read_text()

    # Determine how many particles belong to the first particle population.
    #
    # Because these particles occupy the first consecutive particle IDs,
    # this number also establishes the upper ID of the frozen group.
    frozenC = len(parti1)
    
    # Python list containing the LAMMPS group commands to be inserted.
    #
    # "frozen" contains particle IDs from 1 through frozenC.
    # "mobile" contains every particle not included in "frozen".
    groupF = [
        f"group frozen id 1:{frozenC}",
        "group mobile subtract all frozen"
    ]


    # Optional trajectory-dump command retained for debugging/reference.
    # It is currently disabled and therefore does not affect the simulation.
    #dump_text = 'dump traj all custom 100 output_traj.lammpstrj id type x y z'

    # Construct the block of LAMMPS commands that will be inserted
    # immediately before the simulation begins.
    min_block = [

        # Previously used debugging/reference-output commands.
        # These remain commented out and are not executed.
        #f"dump ref all custom 1 C:/Users/olind/Documents/QuasiCrystalTono/QuasicristalesAlan/Vertices_Teselas/MsdCheck.lammpstrj id type x y z",
        #"run 1",
        #"undump ref",

        # Expand the two group-definition strings created above directly
        # into this list.
        *groupF,

        # Apply zero force in x, y, and z to every particle in the frozen
        # group. This keeps the structure-defining particle population fixed
        # while the mobile particles respond to the simulation conditions.
        "fix freeze frozen setforce 0.0 0.0 0.0",

        # Optional trajectory-output instruction, currently disabled.
        #dump_text,

        # Empty line included in the generated LAMMPS text for separation.
        ""
    ]

    # Join the Python list of LAMMPS commands into a single multiline string.
    block_text = "\n".join(min_block)

    # Locate the first line beginning with the LAMMPS command "run" and insert
    # the group/freeze block immediately before it.
    #
    # count=1 ensures that only the first occurrence of "run" is modified.
    txt = re.sub(r'(?m)^\s*run\b', block_text + "\nrun", txt, count=1)

    # Overwrite the generated LAMMPS input file with the modified version.
    lmpin.write_text(txt)

    return 


def InitFile(init_path):
    """
    Load the previously generated particle structure and separate its two
    particle populations according to the LAMMPS particle-type identifier.

    Particle type 1 and particle type 2 are returned independently as
    coordinate arrays containing only x, y, and z positions.
    """

    # Read the initial structure.
    #
    # finalp is a NumPy array whose columns are:
    # [particle ID, particle type, x, y, z]
    finalp = Readtrj(str(init_path))

    # Select rows corresponding to particle type 1.
    #
    # finalp[:,1] extracts the particle-type column.
    # The Boolean condition == 1 retains only type-1 particles.
    # [:,2:] then removes the ID and type columns, leaving [x, y, z].
    finalp1 = finalp[finalp[:,1] == 1][:,2:]

    # Perform the equivalent filtering operation for particle type 2.
    # The result is another NumPy coordinate array with columns [x, y, z].
    finalp2 = finalp[finalp[:,1] == 2][:,2:]

    # Return the two coordinate datasets separately so that different
    # physical properties can be assigned to each population.
    return finalp1, finalp2


# Structure-dependent spatial scaling parameter.
#
# This value contributes to the dimensions of the simulation region and
# therefore depends on the geometry generated in the preceding structure-
# generation stage.
ScaleV = 6.5
#The scale was also derived from trial error, it can be optimized
# Characteristic spatial scale derived from ScaleV.
rad = 20 * ScaleV


#### Now its the regular run using the points obtained previously during the damp runs and the minimize

# Indicate that the simulation-construction stage has begun.
print("Starting run")


# Define the dimensions of the simulation region as a NumPy array.
#
# The first two components define the x and y dimensions using the
# structure-dependent value 'rad'. The third component gives the
# extent in z. Physical micrometer units are assigned later when
# the magcolloids world is created.
regionReal2 = np.array([rad*1.425, rad*1.425, 4.2])


# Empty Python list used to collect labels identifying the different
# previously generated structures/simulation configurations.
tags = []

# Generate the complete collection of structure identifiers.
#
# i varies from 1 to 10,
# j varies from 1 to 5,
# k takes the two discrete values 1 and 25.
#
# Each combination produces a string of the form:
#     Ne<i>_P<k>_R<j>
#
# These tags are used to distinguish different generated structures and
# their corresponding simulation directories/files.
for i in range(1,11):
    for j in range(1,6):
        for k in (1,25):
            txt = f"Ne{i}_P{k}_R{j}"
            tags.append(txt)


def MultiRun(i,init_path,tag):
    """
    Construct and execute one driven-particle simulation.

    Parameters
    ----------
    i
        Numerical value used here to define the direction of the additional
        magnetic-field tilt and to distinguish individual simulation runs.

    init_path
        Path to the previously generated LAMMPS trajectory/structure that
        supplies the initial particle coordinates.

    tag
        Text identifier describing the generated structure/configuration.
        It is incorporated into the output directory and simulation filename.

    Workflow
    --------
    1. Read and separate the two particle populations.
    2. Assign physical properties to each population.
    3. Define the time-dependent magnetic driving field.
    4. Construct the simulation region/world.
    5. Check whether the requested simulation has already completed.
    6. Generate the LAMMPS input files.
    7. Modify the input so the first particle population remains fixed.
    8. Execute LAMMPS.
    9. Release large simulation objects from memory.
    """

    # Load the initial structure and obtain two NumPy arrays containing
    # the [x, y, z] coordinates of particle types 1 and 2.
    finalp1, finalp2 = InitFile(init_path)

    # Construct the first magcolloids particle population from the
    # type-1 coordinates.
    #
    # Multiplication by ureg.um assigns micrometers as the physical unit
    # of the previously unitless NumPy coordinate array.
    particles = mgc.particles(
        finalp1*ureg.um,

        # Radius of each particle in population 1.
        radius = 1.5*ureg.um,

        # Magnetic susceptibility assigned to this population.
        susceptibility = 0.4,

        # Zero diffusion is assigned to the first population.
        # Together with the later LAMMPS freeze command, this population
        # functions as the fixed part of the generated structure.
        diffusion=0*ureg.um**2/ureg.s,

        # Density parameter used to construct the magcolloids particle object.
        density = 0*ureg.kg/ureg.m**3,

        # Simulation temperature associated with this particle population.
        temperature=300*ureg.K)

    # Construct the second particle population from the type-2 coordinates.
    #
    # Unlike the first population, these particles have a non-zero diffusion
    # coefficient and constitute the mobile population in the driven simulation.
    particles2 = mgc.particles(
        finalp2*ureg.um,

        # Radius of particles in population 2.
        radius = 1.3*ureg.um,

        # Magnetic susceptibility assigned to the mobile particles.
        susceptibility = 0.4,

        # Diffusion coefficient of the mobile particle population.
        diffusion=0.07*ureg.um**2/ureg.s,

        density = 0*ureg.kg/ureg.m**3,
        temperature=300*ureg.K)


    # Define the base time-dependent magnetic driving field.
    #
    # The field has:
    #   magnitude  = 7.2 mT
    #   frequency  = 3 Hz
    #   angle      = 26.9 degrees
    field = mgc.field(magnitude = 7.2*ureg.mT, frequency = 3*ureg.Hz, angle = 26.9*ureg.degrees)

    # Magnitude of the additional field tilt in degrees.
    delta = 7

    # Set the tilt direction using the run-dependent value i.
    deltaDir = i

    # Replace the default magnetic-field components with the tilted
    # time-dependent field defined in redefine_field().
    field = redefine_field(field,delta*ureg.deg,deltaDir*ureg.deg)


    # Construct the physical simulation environment containing both
    # particle populations.
    world = mgc.world([particles,particles2], temperature = 300*ureg.K,

            # Convert the previously defined NumPy region dimensions
            # into physical micrometers.
            region=regionReal2*ureg.um,

            # Boundary conditions in x, y, and z.
            # 'p' denotes periodic boundaries in x and y, while the
            # third direction uses the boundary condition represented by 'm'.
            boundaries = ['p','p','m'],

            # Wall configuration associated with the three dimensions.
            walls = [False,False,True],

            # Maximum distance over which dipolar interactions are evaluated.
            dipole_cutoff = 20*ureg.um)

    # Construct a unique simulation filename from:
    #   structure tag + run/angle identifier.
    #
    # {i:03d} formats the integer using three digits, e.g. 1 -> 001.
    file_name = f"MPV2N13_{tag}_{i:03d}"

    # Build the corresponding output directory.
    #
    # Each structure tag receives its own N13_<tag> directory, and
    # individual runs are separated into run13_<i> subdirectories.
    run_dir = BASE_DIR/f"N13_{tag}"/f"run13_{i:03d}"

    # Before generating or running anything, check whether this exact
    # simulation has already completed successfully.
    #
    # If the completion marker exists in the LAMMPS log, the function exits
    # immediately so the finished simulation is not repeated.
    if lammps_finished_ok(run_dir, file_name):
        return f"Skip {tag} angle={i}"
    

    # Create the directory required for this simulation.
    run_dir.mkdir(parents = True, exist_ok=True)

    # Convert the Path object to a regular string for use by magcolloids.
    dir_name = str(run_dir)
    

    # Configure the magcolloids simulation object.
    sim = mgc.sim(dir_name = dir_name, file_name = file_name, 

            # Numerical integration timestep.
            timestep = 1e-5*ureg.s,

            # Output trajectory sampling rate.
            framerate = 50*ureg.Hz,

            # Total simulated physical time.
            total_time = 15*ureg.s,

            # Include both the fixed structure particles and mobile particles.
            particles = [particles,particles2],

            # Supply the previously configured simulation environment
            # and magnetic driving field.
            world = world, field = field,

            # Store the three spatial coordinates in the simulation output.
            output = ["x","y","z"])


    # Ask magcolloids to generate the files/scripts required by LAMMPS.
    sim.generate_scripts()

    # Construct the path of the newly generated LAMMPS input file.
    inp = Path(sim.dir_name) / f"{sim.file_name}.lmpin"

    # Modify that input file so the first particle population is placed
    # in the frozen group before the LAMMPS run begins.
    #
    # particles.positions supplies the first population whose number is
    # used by RegularRun() to determine the frozen particle-ID range.
    RegularRun(inp,rad, particles.positions)



    # Define the path to the installed LAMMPS executable.
    lmp = str(Path.home() / "src" / "aortiza-lammps" / "build-serial" / "lmp")

    # Define the final LAMMPS input-file path for this particular run.
    lmpin = Path(dir_name) / f"{file_name}.lmpin"

    # Launch LAMMPS as an external process.
    #
    # "-in" supplies the generated input script.
    # cwd sets the run-specific directory as the working directory, keeping
    # the output files associated with this simulation together.
    #
    # check=True causes Python to raise an error if the LAMMPS process
    # terminates unsuccessfully rather than silently continuing.
    subprocess.run(
    [lmp, "-in", str(lmpin)],
    cwd=str(dir_name),
    check=True)
    
    # Explicitly remove the larger simulation objects once the run finishes.
    # This is useful when many independent simulations are executed by a
    # longer batch/cluster workflow.
    del particles, particles2, world, field, sim

    # Request Python garbage collection so memory associated with objects
    # that are no longer referenced can be reclaimed before subsequent runs.
    gc.collect()