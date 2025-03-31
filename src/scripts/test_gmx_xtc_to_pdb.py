import os
import shutil
import tempfile
import subprocess

# Create a temporary directory
temp_dir = tempfile.mkdtemp()
print(f"Created temporary directory: {temp_dir}")

try:
    # Define source and destination paths
    src_gro = "/Volumes/LokeyLabShared/General/ChadsLibrary/Hex/Water/x_258/md_Ref.gro"
    src_xtc = (
        "/Volumes/LokeyLabShared/General/ChadsLibrary/Hex/Water/x_258/300K_fit_4000.xtc"
    )
    src_top = "/Volumes/LokeyLabShared/General/ChadsLibrary/Hex/Water/x_258/topol.top"
    dst_pdb = "/Volumes/LokeyLabShared/General/ChadsLibrary/Hex/Water/x_258/test.pdb"

    # Copy files locally
    local_gro = os.path.join(temp_dir, "md_Ref.gro")
    local_xtc = os.path.join(temp_dir, "300K_fit_4000.xtc")
    local_top = os.path.join(temp_dir, "topol.top")
    local_mdp = os.path.join(temp_dir, "temp.mdp")
    local_tpr = os.path.join(temp_dir, "temp.tpr")
    local_pdb = os.path.join(temp_dir, "test.pdb")

    # Copy structure and trajectory
    print("Copying files...")
    shutil.copy(src_gro, local_gro)
    shutil.copy(src_xtc, local_xtc)
    shutil.copy(src_top, local_top)

    # Create a minimal MDP file for grompp
    print("Creating minimal MDP file...")
    with open(local_mdp, "w") as f:
        f.write("integrator = md\nnsteps = 0\n")

    # Source GROMACS environment if not in PATH
    # Modify this to point to your GROMACS installation
    gmx_cmd = "gmx"  # Default command

    # Check if gmx is in PATH, if not try to find it
    which_gmx = subprocess.run("which gmx", shell=True, capture_output=True, text=True)

    if which_gmx.returncode != 0:
        # Try common GROMACS installation locations
        potential_paths = [
            "/usr/local/gromacs/bin/gmx",
            "/opt/gromacs/bin/gmx",
            os.path.expanduser("~/gromacs/bin/gmx"),
            "/Applications/gromacs/bin/gmx",
            # Add more potential paths here
        ]

        for path in potential_paths:
            if os.path.exists(path):
                gmx_cmd = path
                print(f"Found GROMACS at: {gmx_cmd}")
                break

        # If still not found, try sourcing GMXRC and check again
        if gmx_cmd == "gmx":
            try:
                print("Trying to source GROMACS environment...")
                gmxrc_paths = [
                    "/usr/local/gromacs/bin/GMXRC",
                    "/opt/gromacs/bin/GMXRC",
                    os.path.expanduser("~/gromacs/bin/GMXRC"),
                    "/Applications/gromacs/bin/GMXRC",
                ]

                for gmxrc in gmxrc_paths:
                    if os.path.exists(gmxrc):
                        # Source GMXRC and update environment
                        source_cmd = f"source {gmxrc} && which gmx"
                        result = subprocess.run(
                            source_cmd,
                            shell=True,
                            executable="/bin/bash",
                            capture_output=True,
                            text=True,
                        )
                        if result.returncode == 0 and result.stdout.strip():
                            gmx_cmd = result.stdout.strip()
                            print(f"GROMACS sourced from {gmxrc}. Using: {gmx_cmd}")
                            break
            except Exception as e:
                print(f"Error sourcing GROMACS: {e}")

    print(f"Using GROMACS command: {gmx_cmd}")

    # Use GROMACS to create a TPR file and then convert to PDB with CONECT records
    print("Running GROMACS commands...")

    # Create TPR file
    print("Creating TPR file...")
    cmd_grompp = f"{gmx_cmd} grompp -f {local_mdp} -c {local_gro} -p {local_top} -o {local_tpr} -maxwarn 100"
    print(f"Command: {cmd_grompp}")
    result = subprocess.run(cmd_grompp, shell=True, capture_output=True, text=True)
    print(f"Stdout: {result.stdout}")
    print(f"Stderr: {result.stderr}")
    print(f"Return code: {result.returncode}")

    if result.returncode != 0:
        raise Exception("Failed to create TPR file")

    # Extract first frame with CONECT records
    print("Converting to PDB with CONECT records...")
    cmd_trjconv = f"echo 0 | {gmx_cmd} trjconv -s {local_tpr} -f {local_xtc} -o {local_pdb} -pbc mol -conect"
    print(f"Command: {cmd_trjconv}")
    result = subprocess.run(cmd_trjconv, shell=True, capture_output=True, text=True)
    print(f"Stdout: {result.stdout}")
    print(f"Stderr: {result.stderr}")
    print(f"Return code: {result.returncode}")

    if result.returncode != 0:
        raise Exception("Failed to convert to PDB")

    # Copy result back to shared drive
    print("Copying result back to shared drive...")
    shutil.copy(local_pdb, dst_pdb)
    print("Done!")

    # Check if CONECT records were written
    with open(local_pdb, "r") as f:
        content = f.read()
        conect_lines = [
            line for line in content.split("\n") if line.startswith("CONECT")
        ]
        print(f"Number of CONECT records written: {len(conect_lines)}")
        if len(conect_lines) > 0:
            print("Sample CONECT records:")
            for line in conect_lines[:5]:  # Show first 5 CONECT records
                print(line)

except Exception as e:
    print(f"Error: {e}")

finally:
    # Clean up
    print("Cleaning up temporary files...")
    shutil.rmtree(temp_dir)
    print("Temporary directory removed")
