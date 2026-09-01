
import os
import sys
from pathlib import Path
import re
import subprocess

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Determine the project root relative to the location of this script and
# temporarily add it to Python's module-search path. This allows the local
# magcolloids package/repository to be imported from the project structure.
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
import magcolloids as mgc
from IPython.display import HTML

ureg = mgc.ureg

# pandas indexing helper retained for indexed/sliced data operations.
idx = pd.IndexSlice


## Funcs y cosas para despues

def redefine_field(field,tilt_angle,tilt_dir):
    """
    Redefine the components of the applied magnetic field to introduce
    an additional field tilt.

    The supplied magcolloids field already contains the base magnetic-field
    magnitude, frequency, and angular parameter. This function calculates
    the geometric projections associated with an additional tilt angle and
    tilt direction and incorporates them into the time-dependent x, y, and z
    field expressions used by LAMMPS.
    """

    # Calculate the x-oriented projection associated with the imposed tilt.
    # The unit-aware angular quantities are converted to their numerical
    # magnitudes and then from degrees to radians.
    sinTAcosTD = np.sin(np.deg2rad(tilt_angle.magnitude))*np.cos(np.deg2rad(tilt_dir.magnitude))

    # Calculate the corresponding y-oriented projection.
    sinTAsinTD = np.sin(np.deg2rad(tilt_angle.magnitude))*np.sin(np.deg2rad(tilt_dir.magnitude))

    # Construct the time-dependent x component of the magnetic field.
    # The expression is stored as a string so it can be incorporated into
    # the generated LAMMPS input.
    field.fieldx = f'v_Bmag*(sin(v_freq*time*2*PI)*sin(v_theta)+cos(v_theta)*{sinTAcosTD})'

    # Construct the time-dependent y component.
    field.fieldy = f'v_Bmag*(cos(v_freq*time*2*PI)*sin(v_theta)+cos(v_theta)*{sinTAsinTD})'

    # Construct the vertical (z) component of the tilted field.
    field.fieldz = f'v_Bmag*(cos(v_theta)*{np.cos(np.deg2rad(tilt_angle.magnitude))})'

    # Return the field object with its components redefined.
    return field


def minimizer(lmpin: Path, parti1,i,j, k_wall=100):
    """
    Modify the generated LAMMPS input to prepare and relax the initial
    two-population particle configuration.

    The type-1 particles define the fixed quasiperiodic structure and are
    frozen during this procedure. The remaining particles form the mobile
    population.

    Before the main simulation, the mobile population undergoes two strongly
    damped runs followed by FIRE energy minimization. Existing LAMMPS fixes
    that could interfere with this preparation stage are temporarily removed
    and restored afterward.

    Parameters i and j identify the particular generated configuration/run.
    """

    import re

    # Read the complete generated LAMMPS input file into a Python string.
    txt = lmpin.read_text()

    # Catch any existing fix lines that could interfere
    #
    # Search the generated LAMMPS input for existing "fix" commands involving
    # selected integration, forcing, thermostat, or field operations.
    #
    # These fixes are temporarily suspended while the custom damping and
    # minimization sequence is performed.
    pat = r'(?m)^(fix\s+(\S+)\s+.*\b(setdipole|bd|langevin|addforce|efield|nve(?:/limit)?|nvt|npt|move)\b.*)$'
    susp = list(re.finditer(pat, txt))

    # Generate one "unfix" command for every detected LAMMPS fix.
    unfix_lines   = "\n".join(f"unfix {m.group(2)}" for m in susp)

    # Preserve the original fix commands so they can be restored after
    # the relaxation/minimization procedure.
    restore_lines = "\n".join(m.group(1)          for m in susp)

    # Determine the number of particles belonging to the fixed
    # quasiperiodic population.
    frozenC = len(parti1)

    # Define two LAMMPS particle groups:
    #
    # frozen -> type-1 structural particles, assumed to occupy IDs 1:frozenC
    # mobile -> every particle not belonging to the frozen group
    groupF = [
        f"group frozen id 1:{frozenC}",
        "group mobile subtract all frozen"
    ]

    
   # Debug print to catch radial positions

    # Debugging commands used to inspect the radial position of mobile
    # particles during development.
    #
    # The first expression calculates each particle's radial distance
    # sqrt(x² + y² + z²), while the second would output those values
    # alongside the particle coordinates.
    debug_print = [
        "variable radatom atom sqrt(x*x+y*y+z*z)",
        'dump wallcheck mobile custom 1 wall_debug.txt id type x y z v_radatom'
    ]

    ## The following instructions were made based on trial and error, it probably ISNT the most optimized version but for the ranges used
    #it works, this can be checked in ovito to see if in the end script there isnt any overlap o simply check the log for errors
    # !!! THE FILE WILL GENERATE EVEN IF THERE ARE ERRORS. So file generation doesnt mean the overlaps and initial state errors are necesarilly fixed
    #Finally take into account that the original file is generated randomly, which means that it could be that sometimes this protocol isnt enough
    # but usually it works, if it doesnt work the first time just rerun it

    # Construct the sequence of LAMMPS commands used to prepare and
    # relax the initial configuration.
    min_block = [

        # Temporarily remove potentially interfering fixes found in the
        # originally generated LAMMPS input.
        unfix_lines,

        # Insert the frozen/mobile particle-group definitions.
        *groupF,

        # Set all force components on the quasiperiodic structural
        # population to zero, keeping these particles fixed.
        "fix freeze frozen setforce 0.0 0.0 0.0",

        # Save the original particle configuration before relaxation.
        # Each row of the trajectory contains:
        # particle ID, type, x, y, and z coordinates.
        "dump original all custom 1 origi.lammpstrj id type x y z",

        # Advance the simulation by one step so the original configuration
        # is written to the trajectory dump.
        "run 1",

        # Stop writing the original-configuration dump.
        "undump original",

        # FIRST DAMPING STAGE:
        # Apply very strong viscous damping to the mobile population.
        "fix pre_damp mobile viscous 1e6", #Damp run 1

        # Limit the maximum displacement produced during integration,
        # helping stabilize the initially generated mobile configuration.
        "fix nvel mobile nve/limit 1e-6",

        # Use a timestep of 1e-3 during the first damping stage.
        "timestep 1e-3",

        # Perform 2000 damping steps.
        "run 2000",

        # Remove the first-stage damping and integration constraints.
        "unfix pre_damp",
        "unfix nvel",

        # SECOND DAMPING STAGE:
        # Repeat the relaxation with a lower viscous damping coefficient.
        "fix pre_damp mobile viscous 1e3", #Damp run 2

        # Again constrain particle motion during integration.
        "fix nvel mobile nve/limit 1e-6",

        # Increase the timestep for the second damping stage.
        "timestep 1e-2",

        # Perform another 2000 damping steps.
        "run 2000",

        # Remove the second-stage damping constraints.
        "unfix pre_damp",
        "unfix nvel",

        # ENERGY-MINIMIZATION STAGE:
        # Select the FIRE (Fast Inertial Relaxation Engine)
        # minimization algorithm.
        "min_style fire", #Min

        # Restrict the maximum displacement during minimization.
        "min_modify dmax 0.01",

        # Minimize the system using the specified energy tolerance,
        # force tolerance, maximum iteration count, and evaluation limit.
        "minimize 1e-6 1e-7 70000 70000",

        # Restore the LAMMPS fixes that were suspended before relaxation.
        restore_lines,

        # Continue the system for an additional 6000 simulation steps
        # after restoring the original simulation fixes.
        "run 6000",

        # Previously used output commands for saving a final configuration.
        # These remain disabled and do not affect the current execution.
        #f"dump final all custom 1 C:/Users/olind/Documents/QuasiCrystalTono/QuasicristalesAlan/Vertices_Teselas2/start_Ne{i}_P1_R{j}.lammpstrj id type x y z",
        #"run 1",
        #"undump final",

        ""
    ]

    # Build minimization block

    # Join the individual LAMMPS instructions into one multiline block.
    block_text = "\n".join(min_block)


    # Replace first run command with the debug minimization

    # Insert the complete preparation/minimization block immediately before
    # the first "run" instruction in the generated LAMMPS input.
    #
    # count=1 ensures that only the first run command is modified.
    txt = re.sub(r'(?m)^\s*run\b', block_text + "\nrun", txt, count=1)

    # Overwrite the generated input file with the modified version.
    lmpin.write_text(txt)


def Readtrj(trjfile):
    """
    Read particle records from a LAMMPS trajectory file.

    Each particle is represented by:
        [particle ID, particle type, x, y, z]

    The complete collection is returned as a NumPy array.
    """

    # Initialize a Python list that will accumulate particle records.
    coord = []

    # Load the trajectory file as individual text lines.
    with open(trjfile) as f:
        lines = f.readlines()

    # Search for the LAMMPS "ITEM: ATOMS" header.
    # If multiple ATOMS blocks are present, 'start' ultimately corresponds
    # to the last block encountered in the file.
    for i, line in enumerate(lines):
        if line.startswith("ITEM: ATOMS"):
            start = i +1

    # Read the particle records following the selected ATOMS header.
    for line in lines[start:]:

        # Split the whitespace-separated LAMMPS values.
        parts = line.split()

        # Particle ID and particle type are interpreted as integers.
        atomid = int(parts[0])
        atype = int(parts[1])

        # Particle positions are interpreted as floating-point coordinates.
        x, y, z = map(float, parts[2:5])

        # Append one complete particle record.
        coord.append([atomid, atype, x, y, z])

    # Convert the list of records into a numerical NumPy array.
    return np.array(coord)


def RegularRun(lmpin: Path,rad):
    """
    Modify a LAMMPS input for a regular simulation run.

    The fixed quasiperiodic particles are grouped as 'frozen', while the
    remaining particles are assigned to the 'mobile' group. The frozen
    population receives zero force and a trajectory dump is enabled.

    This helper is defined in this generation script but is separate from
    the minimizer() preparation procedure used below.
    """

    import re

    # Read the generated LAMMPS input file.
    txt = lmpin.read_text()

    # Number of particles belonging to the fixed quasiperiodic structure.
    frozenC = len(parti1)
    
    # Define the fixed and mobile LAMMPS groups.
    groupF = [
        f"group frozen id 1:{frozenC}",
        "group mobile subtract all frozen"
    ]

  

    # LAMMPS command for periodically saving particle IDs, types,
    # and three-dimensional coordinates.
    dump_text = 'dump traj all custom 100 output_traj.lammpstrj id type x y z'

    # Commands inserted before the first simulation run.
    min_block = [
        *groupF,

        # Keep the quasiperiodic structural population fixed.
        "fix freeze frozen setforce 0.0 0.0 0.0",

        # Save trajectory data every 100 simulation steps.
        dump_text,
        ""
    ]

    # Convert the list of commands into a multiline string.
    block_text = "\n".join(min_block)

    # Insert the commands before the first LAMMPS run instruction.
    txt = re.sub(r'(?m)^\s*run\b', block_text + "\nrun", txt, count=1)

    # Save the modified input file.
    lmpin.write_text(txt)



# Generate multiple initial particle configurations.
#
# i selects one of the quasiperiodic vertex datasets.
for i in range(1,11):

    # Development filters retained from previous testing.
    # They are disabled, so every value of i is currently processed.
    #if i != 10:
        #continue

    # j provides five independent realization/run identifiers for each
    # quasiperiodic structure.
    for j in range(1,6):

        # Additional development filter retained but currently disabled.
        #if j != 1:
         #   continue

        # Spatial scaling factor applied to the original quasiperiodic
        # vertex coordinates.
        ScaleV = 6.5 #Og was 6.5

        # Characteristic system size derived from the scaling factor.
        rad = 20 * ScaleV

        # Paths de linux (lammps_param es el clon del repositorio)

        # Path to the previously generated quasiperiodic tiling vertices.
        #
        # The particular input filename depends on i, so each iteration
        # loads a different generated structure.
        vertices_file = Path(f"C:/Users/olind/Documents/QuasiCrystalTono/QuasicristalesAlan/Vertices_Teselas2/Olin_2D_Quasiperiodic_Tiling/N13/Vertices_Teselas_N13_R40_{i}.csv")

        # Output path retained for this generation workflow.
        output_file = Path("C:/Users/olind/Documents/QuasiCrystalTono/QuasicristalesAlan/Vertices_Teselas2/Tester.csv")

        #partis 1 y 2 

        # -----------------------------------------------------------------
        # PARTICLE POPULATION 1: QUASIPERIODIC STRUCTURE
        # -----------------------------------------------------------------

        # Load the quasiperiodic vertex coordinates into a pandas DataFrame.
        #
        # The input contains no header row and uses tab-separated values.
        Data = pd.read_csv(vertices_file, header=None, delimiter="\t")

        # Explicitly represent the imported data as a pandas DataFrame.
        Data = pd.DataFrame(Data)

        # CLEANING:
        # Remove duplicate quasiperiodic vertices based on their x and y
        # coordinate columns. This prevents multiple fixed particles from
        # being created at the same planar location.
        Data = Data.drop_duplicates(subset = [0,1])

        # Extract the x and y coordinate columns as pandas Series.
        x = Data[0]
        y = Data[1]

        # Convert the planar quasiperiodic vertices into an N x 3 NumPy
        # coordinate array.
        #
        # Each row has the form:
        #     [x, y, 1]
        #
        # so all particles in this population initially share the same
        # z-coordinate.
        parti1 = np.column_stack([x, y, np.ones(len(x))])

        # SPATIAL FILTERING / CROPPING:
        # Construct a Boolean mask that retains only quasiperiodic vertices
        # inside the square:
        #
        #   -10√2 < x < 10√2
        #   -10√2 < y < 10√2
        #
        # This extracts the desired central region from the larger generated
        # quasiperiodic tiling.

        #Size is choosen based on desired simulation spaces, it can be adjusted but physically this will change the box size and more things will
        #need to be adjusted

        mask1 = (
            (parti1[:,0] > -10*np.sqrt(2)) &
            (parti1[:,0] <  10*np.sqrt(2)) &
            (parti1[:,1] > -10*np.sqrt(2)) &
            (parti1[:,1] <  10*np.sqrt(2))
        )

        # Apply the spatial mask to the structural-particle coordinate array.
        # parti1 now contains only vertices within the selected region.
        parti1 = parti1[mask1]

        # Count the number of retained quasiperiodic structural particles.
        Len= len(parti1[:,0])

        # Determine the number of particles in the second population.
        #
        # Here, the mobile population is set to 25% of the number of
        # retained quasiperiodic structural particles, rounded to the
        # nearest integer.
        PercentageLen = int(np.round(0.25*Len))
        
        #The ratio from fixed to loose was chosen after trying with different ratios, the smaller the ratio the better notion you get of how free particles
        # interact with the substrate. however if you dont have enough you are in danger of random generation not being averaged over and it starting
        # to look more like a single particle sim, which can sometimes move a lot and sometimes not move at all

        # -----------------------------------------------------------------
        # PARTICLE POPULATION 2: MOBILE PARTICLES
        # -----------------------------------------------------------------

        # Define the half-width of the region used for random mobile-particle
        # placement before the global spatial scaling is applied.
        L = (20/2)*np.sqrt(2)

        # Generate PercentageLen random x coordinates from a uniform
        # distribution spanning approximately 99% of the available region.
        x2 = np.random.uniform(-L*0.99, L*0.99, PercentageLen)

                # Generate PercentageLen random y coordinates from the same
        # uniform spatial distribution used for the x coordinates.
        #
        # Together, x2 and y2 define the random initial planar positions
        # of the mobile-particle population.
        y2 = np.random.uniform(-L*0.99, L*0.99, PercentageLen)

        # Build the N x 3 coordinate array for particle population 2.
        #
        # Each row has the form:
        #     [x, y, 0.6]
        #
        # The x and y positions are randomly generated, while every
        # mobile particle is initially assigned the same z-coordinate.
        parti2 = np.column_stack([x2, y2, np.full(PercentageLen, 0.6)])

        #The z coordinate of 0.6 was generated through trial and error finding at which spot the particles
        # showed the desired dance like motion
        #It can be modified and improved



        # -----------------------------------------------------------------
        # SPATIAL SCALING
        # -----------------------------------------------------------------

        # Apply the global spatial scale factor to the x coordinates
        # of the fixed quasiperiodic particle population.
        parti1[:,0] *= ScaleV

        # Apply the same scaling to the y coordinates of the fixed
        # quasiperiodic population.
        parti1[:,1] *= ScaleV

        # Scale the x coordinates of the randomly generated mobile
        # particles so that both particle populations occupy the same
        # physical coordinate system.
        parti2[:,0] *= ScaleV

        # Scale the y coordinates of the mobile particles.
        parti2[:,1] *= ScaleV


        #Moving partis 
        # Combine positions

        # Stack the fixed quasiperiodic particles and the mobile particles
        # vertically into a single NumPy array.
        #
        # The resulting array contains the coordinates of every particle
        # in the generated initial configuration.
        #
        # This combined array is useful for inspecting the total system
        # size or particle count, although the two populations are still
        # passed separately to magcolloids below because they have different
        # physical properties.
        all_particles = np.vstack([parti1, parti2])

        # Print the total number of particles in the generated system.
        print(len(all_particles))


        # Define the dimensions of the simulation region.
        #
        # regionReal is a three-element NumPy array describing the system
        # dimensions before physical micrometer units are attached.
        #
        # The x and y dimensions depend on the structure-dependent value
        # 'rad', while the z dimension is fixed at 4.2.
        regionReal = np.array([rad*1.425, rad*1.425, 4.2])


        # -----------------------------------------------------------------
        # PARTICLE POPULATION 1: FIXED QUASIPERIODIC STRUCTURE
        # -----------------------------------------------------------------

        # Create the magcolloids object corresponding to the structural
        # quasiperiodic particle population.
        #
        # The coordinate NumPy array is multiplied by ureg.um so that
        # magcolloids interprets the particle positions in micrometers.
        particles = mgc.particles(
            parti1*ureg.um,

            # Radius assigned to each structural particle.
            radius = 1.5*ureg.um,

            # Magnetic susceptibility of the structural particles.
            susceptibility = 0.4,

            # Zero diffusion coefficient is assigned to this population.
            # These particles are subsequently frozen during the relaxation
            # procedure and therefore define the stationary quasiperiodic
            # structure over which the mobile particles evolve.
            diffusion=0*ureg.um**2/ureg.s,

            # Density parameter used to construct the particle object.
            density = 0*ureg.kg/ureg.m**3,

            # System temperature associated with this population.
            temperature=300*ureg.K)


        # -----------------------------------------------------------------
        # PARTICLE POPULATION 2: MOBILE PARTICLES
        # -----------------------------------------------------------------

        # Create the magcolloids object corresponding to the randomly
        # initialized mobile-particle population.
        particles2 = mgc.particles(
            parti2*ureg.um,

            # Mobile particles use a slightly smaller radius than the
            # particles forming the fixed quasiperiodic structure.
            radius = 1.3*ureg.um,

            # Magnetic susceptibility of the mobile particles.
            susceptibility = 0.4,

            # Non-zero diffusion coefficient assigned to the mobile
            # population, allowing Brownian/diffusive motion in the
            # subsequent dynamics.
            diffusion=0.07*ureg.um**2/ureg.s,

            density = 0*ureg.kg/ureg.m**3,
            temperature=300*ureg.K)


        # -----------------------------------------------------------------
        # MAGNETIC DRIVING FIELD
        # -----------------------------------------------------------------

        # Define the base time-dependent magnetic field used during
        # the simulation.
        #
        # Field parameters:
        #     magnitude = 7.2 mT
        #     frequency = 3 Hz
        #     angle     = 26.9 degrees
        field = mgc.field(magnitude = 7.2*ureg.mT, frequency = 3*ureg.Hz, angle = 26.9*ureg.degrees)

        # Magnitude of the additional magnetic-field tilt.
        delta = 7

        # For this initial-generation/relaxation stage, the tilt direction
        # is fixed at zero degrees.
        deltaDir = 0

        # Replace the default field components with the tilted field
        # expressions defined in redefine_field().
        field = redefine_field(field,delta*ureg.deg,deltaDir*ureg.deg)



        # -----------------------------------------------------------------
        # SIMULATION ENVIRONMENT
        # -----------------------------------------------------------------

        # Construct the magcolloids simulation world containing both
        # the fixed quasiperiodic structure and the mobile particles.
        world = mgc.world([particles,particles2], temperature = 300*ureg.K,

                        # Convert the previously calculated region dimensions
                        # to physical micrometer units.
                        region=regionReal*ureg.um,

                        # Define the boundary conditions in x, y, and z.
                        # In this generation/relaxation stage all three
                        # directions use the boundary condition represented
                        # by 'f'.
                        boundaries = ['f','f','f'],

                        # Enable a wall only in the third spatial direction.
                        walls = [False,False,True],

                        # Interaction cutoffs used when building the system.
                        dipole_cutoff = 5*ureg.um, lj_cutoff = 5,

                        # Lennard-Jones interaction parameters supplied to
                        # the magcolloids world.
                        lj_parameters = [0.01*ureg.pg*ureg.um**2/ureg.us**2, 0.8908987181403393])


        # -----------------------------------------------------------------
        # INITIAL RELAXATION SIMULATION
        # -----------------------------------------------------------------

        # Construct the magcolloids simulation object.
        #
        # The output filename contains:
        #     N13        -> structure/system identifier
        #     Ne{i}      -> selected quasiperiodic realization
        #     P25        -> 25% mobile-particle population
        #     R{j}       -> realization/run index
        #
        # The exact naming convention is specific to the generated
        # structure and project organization.
        sim = mgc.sim(dir_name =str(vertices_file.parent), file_name = f"N13_start_Ne{i}_P25_R{j}",

            # Base timestep passed to the generated simulation.
            timestep = 1e-5*ureg.s,

            # Requested trajectory/output frame rate.
            framerate = 30*ureg.Hz,

            # Nominal simulation duration supplied to magcolloids.
            # The generated LAMMPS input is subsequently modified by
            # minimizer() to insert the custom staged relaxation protocol.
            total_time = 0.05*ureg.s,

            # Pass both particle populations to the simulation.
            particles = [particles,particles2],

            # Use the previously constructed simulation environment
            # and magnetic field.
            world = world, field = field,

            # Store the three particle-coordinate components in the output.
            output = ["x","y","z"])



        # Generate the LAMMPS scripts and input files corresponding to
        # the magcolloids simulation configuration.
        sim.generate_scripts()

        # Construct the path to the newly generated LAMMPS input file.
        inp = Path(sim.dir_name) / f"{sim.file_name}.lmpin"

        # Modify the generated input file by inserting the custom staged
        # initialization protocol:
        #
        #     frozen substrate
        #          ↓
        #     strong damping
        #          ↓
        #     reduced damping
        #          ↓
        #     FIRE minimization
        #          ↓
        #     restored simulation fixes / final relaxation
        #
        # parti1 is passed so that minimizer() can determine how many
        # particles belong to the fixed quasiperiodic population.
        minimizer(inp, parti1,i,j, k_wall=10)

        # Execute the simulation using the modified LAMMPS input.
        #
        # At this point, the initially random mobile-particle configuration
        # is subjected to the complete damping/minimization procedure.
        sim.run()

        # Count the number of particles belonging to the fixed
        # quasiperiodic population.
        LenP1 = len(parti1)

        # Print this particle count for inspection/logging.
        print(LenP1)

        # Retrieve the simulation-region representation stored by the
        # magcolloids world object.
        size = world.region

        # Calculate the simulation-box volume from the corresponding
        # region limits stored in world.region.
        #
        # The factor of 8 accounts for the use of half-extent values in
        # the three spatial directions.
        Bvol = size[1]*size[3]*size[5]*8

        # Estimate the total particle volume using the spherical-particle
        # volume expression:
        #
        #     V = (4/3) * pi * r^3
        #
        # The calculation combines contributions based on the radii
        # associated with the two particle populations.
        Pvol = (4/3)*(np.pi)*((1.5**3)*particles.positions.shape[0] + (1.3**3)*particles.positions.shape[0])

        # Calculate the particle packing fraction as the ratio between
        # the estimated total particle volume and simulation-box volume.
        packing = Pvol/Bvol

        # Print the resulting packing estimate.
        print(packing)