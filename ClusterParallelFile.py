# Standard-library modules used to access environment variables
# and construct filesystem paths in a platform-independent form.
import os
from pathlib import Path


# ---------------------------------------------------------------------------
# CPU THREAD CONTROL
# ---------------------------------------------------------------------------
#
# Restrict numerical libraries to a single computational thread per Python
# process.
#
# This script already parallelizes simulations using multiprocessing.Pool.
# Libraries such as NumPy may internally use OpenMP, MKL, OpenBLAS, or
# NumExpr threads. Without these limits, every worker process could create
# additional threads, causing CPU oversubscription on the cluster.
#
# Setting each library to one thread therefore allows the number of active
# simulation workers to remain consistent with the CPUs allocated by SLURM.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"


# Pool provides process-based parallelism within each SLURM array task.
# freeze_support() provides compatibility with multiprocessing initialization,
# particularly when Python uses process-spawning behavior.
from multiprocessing import Pool, freeze_support

# Import the production simulation module.
#
# This module contains:
#   - the complete list of generated structure tags (allp.tags)
#   - MultiRun(), which constructs and executes one LAMMPS simulation
#     for a particular structure and magnetic-field direction.
import SimulationFile.py as allp


def safe(args):
    """
    Execute one production simulation while capturing any errors.

    Parameters
    ----------
    args : tuple
        Contains:
            angle         -> magnetic-field tilt direction for this run
            init_path_str -> path to the relaxed initial particle configuration
            tag           -> identifier of the quasiperiodic structure

    Returns
    -------
    tuple
        Successful simulation:
            ("ok", angle, result)

        Failed simulation:
            ("fail", angle, traceback)

    Wrapping MultiRun() in this function prevents one failed simulation
    from immediately terminating the complete multiprocessing batch.
    """

    # Unpack the three pieces of information associated with one
    # independent simulation task.
    angle, init_path_str, tag = args

    try:
        # Execute one complete production simulation.
        #
        # The initial configuration path was converted to a string before
        # being passed to the worker process, so it is reconstructed here
        # as a pathlib Path object.
        res = allp.MultiRun(angle, Path(init_path_str), tag)

        # Return a structured success result containing the angle and
        # any message returned by MultiRun().
        return ("ok", angle, res)

    except Exception:
        # If the simulation fails, capture the complete Python traceback
        # instead of allowing the worker exception to terminate the
        # remaining simulations.
        import traceback

        # Return the failure status, corresponding field angle, and
        # formatted traceback for later reporting.
        return ("fail", angle, traceback.format_exc())



# ---------------------------------------------------------------------------
# MAIN CLUSTER EXECUTION
# ---------------------------------------------------------------------------
#
# This block is executed only when this file is launched directly.
# It is not executed if the file is imported as a Python module.
if __name__ == "__main__":

    # Initialize multiprocessing support before worker processes are created.
    freeze_support()


    # -----------------------------------------------------------------------
    # SLURM RESOURCE INFORMATION
    # -----------------------------------------------------------------------

    # Read the SLURM array-task identifier from the cluster environment.
    #
    # Each SLURM array task receives a different integer ID and therefore
    # processes a different subset of the generated quasiperiodic structures.
    #
    # The default value "0" allows the script to behave as task 0 if the
    # environment variable is not present.
    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))   # 0..19

    # Read the number of CPUs allocated to this SLURM task.
    #
    # This value determines the number of Python worker processes that
    # will be created simultaneously below.
    #
    # If SLURM_CPUS_PER_TASK is unavailable, eight workers are used.
    nprocs  = int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))


    # -----------------------------------------------------------------------
    # DISTRIBUTE STRUCTURES ACROSS SLURM ARRAY TASKS
    # -----------------------------------------------------------------------

    # Number of quasiperiodic structure tags assigned to each SLURM
    # array task.
    tags_per_job = 5

    # Calculate the first tag index assigned to the current array task.
    #
    # Example:
    #   task 0 -> start = 0
    #   task 1 -> start = 5
    #   task 2 -> start = 10
    start = task_id * tags_per_job

    # Calculate the exclusive upper index of the assigned tag range.
    end = start + tags_per_job

    # Slice the complete tag list defined in the production simulation
    # module to obtain only the structures assigned to this SLURM task.
    #
    # my_tags is therefore a Python list containing at most five
    # structure-identification strings.
    my_tags = allp.tags[start:end]


    # Verify that the current SLURM task was assigned at least one
    # valid structure.
    #
    # An empty slice generally indicates that the requested array-task
    # index exceeds the available number of structure tags.
    if len(my_tags) == 0:
        raise RuntimeError(f"No tags for task_id={task_id} (start={start}, end={end})")


    # -----------------------------------------------------------------------
    # MAGNETIC-FIELD DIRECTIONS
    # -----------------------------------------------------------------------

    # Generate the complete set of field-direction values to simulate.
    #
    # range(0, 365, 5) produces:
    #
    #     0, 5, 10, 15, ..., 360
    #
    # Each value is passed to MultiRun() as the direction of the additional
    # magnetic-field tilt.
    #
    # Consequently, every initial particle configuration is simulated
    # independently across the complete angular sweep in 5-degree increments.
    jobs = list(range(0, 365, 5))

    # Store the number of angular simulations performed for each structure.
    # This value is used when reporting execution progress.
    total = len(jobs)


    # -----------------------------------------------------------------------
    # INITIAL-CONFIGURATION DIRECTORY
    # -----------------------------------------------------------------------

    # Obtain the project base directory from the cluster environment.
    #
    # Unlike the production module, this orchestration script requires
    # BASE_DIR to exist in the environment; no fallback directory is
    # supplied here.
    BASE_DIR = Path(os.environ["BASE_DIR"]).resolve()

    # The relaxed/minimized configurations produced during the generation
    # stage are expected to be stored in the "init_configs" subdirectory.
    INIT_DIR = BASE_DIR / "init_configs"


    # Report which portion of the complete structure dataset has been
    # assigned to this SLURM array task.
    #
    # flush=True forces the message to appear immediately in the cluster
    # output log rather than remaining in Python's output buffer.
    print(f"[task {task_id}] tags {start}..{end-1} -> {my_tags}", flush=True)


    # -----------------------------------------------------------------------
    # PROCESS EACH ASSIGNED QUASIPERIODIC STRUCTURE
    # -----------------------------------------------------------------------

    # Iterate sequentially through the structure tags assigned to this
    # SLURM array task.
    #
    # The structures themselves are processed one at a time here.
    # Parallelization occurs across the different field angles for each
    # individual structure.
    for tag in my_tags:

        # Construct the expected path of the relaxed initial configuration
        # generated during the previous simulation-preparation stage.
        #
        # Each file corresponds to one particular quasiperiodic structure
        # and mobile-particle realization.
        init_path = INIT_DIR / f"N13_start_{tag}.lammpstrj"


        # Confirm that the required initial particle configuration exists
        # before attempting to launch its production simulations.
        #
        # Missing initial data causes an explicit error rather than allowing
        # dozens of worker processes to fail independently.
        if not init_path.exists():
            raise FileNotFoundError(f"Missing init file: {init_path}")


        # Write the current structure, initial-configuration path, and
        # number of allocated worker processes to the SLURM output log.
        print(f"[task {task_id}] Running tag={tag} init={init_path} with {nprocs} procs", flush=True)


        # -------------------------------------------------------------------
        # BUILD THE ANGULAR SIMULATION TASKS
        # -------------------------------------------------------------------

        # Create one tuple for every magnetic-field direction.
        #
        # Each tuple contains:
        #
        #     (field angle, initial configuration path, structure tag)
        #
        # All angular simulations use the same initial particle configuration
        # for the current tag. Only the field direction changes between tasks.
        #
        # The Path is converted to a string before being passed between
        # multiprocessing workers.
        tasks = [(angle, str(init_path), tag) for angle in jobs]


        # -------------------------------------------------------------------
        # PARALLEL EXECUTION WITHIN THE SLURM TASK
        # -------------------------------------------------------------------

        # Create a multiprocessing pool containing one worker process per
        # allocated CPU, as determined by SLURM_CPUS_PER_TASK.
        #
        # maxtasksperchild=50 causes a worker process to be replaced after
        # it has processed 50 tasks. For repeated simulation workloads,
        # periodically replacing workers can help prevent long-term memory
        # accumulation from affecting the complete batch.
        with Pool(processes=nprocs, maxtasksperchild=50) as p:

            # Execute the angular simulations in parallel.
            #
            # imap_unordered() returns completed simulations as soon as they
            # finish rather than waiting for them in angular order. This is
            # useful because different LAMMPS simulations may require
            # different amounts of execution time.
            #
            # chunksize=1 assigns individual angular simulations to workers
            # independently rather than grouping several angles into a
            # single multiprocessing task.
            #
            # enumerate(..., start=1) supplies a running completion counter.
            for i, (status, ang, msg) in enumerate(p.imap_unordered(safe, tasks, chunksize=1), start=1):

                # Report successful completion of an angular simulation.
                if status == "ok":
                    print(f"{i}/{total}: Finished angle {ang} → {msg}", flush=True)

                # If safe() captured an exception, report the corresponding
                # angle together with its complete traceback.
                else:
                    print(f"{i}/{total}: ERROR angle {ang}\n{msg}", flush=True)

                # Print the current completion count separately.
                print(i)