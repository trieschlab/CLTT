#!/bin/bash 
# _____________________________________________________________________________
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=1 
#SBATCH --cpus-per-task=4 
#SBATCH --time=48:00:00 
#SBATCH --mem=30GB 
#SBATCH --gres=gpu:1 
#SBATCH --partition=sleuths 
#SBATCH --job-name=experiment_name
#SBATCH --mail-type=END 
#SBATCH --mail-user=username@fias.uni-frankfurt.de 
#SBATCH --output=/path/to/repo/CLTT/save/slurm_%j.out 
#SBATCH --array=0%1 
#SBATCH --reservation triesch-shared 

# order of array
j=$((SLURM_ARRAY_TASK_ID))

nfix_array=(2 5 10)



echo "job $j"	

srun python3 main/train.py \
	--n_fix ${nfix_array[$j]} \



# --- end of experiment ---

# ----------------
# read before using!
# ----------------

# custom user preferences
# -----

# duplicate this file in your dev_branch (i.e. run_on_cluster.sbatch) 
# and make it your own.
# Do NOT edit this in main.

# Adapt the following parameters at the top of the document
# --job-name=experiment_name
# --mail-user=username@fias.uni-frankfurt.de 
# --output=/path/to/repo/CLTT/save/slurm_%j.out 


# experiment preferences
# -----

# if you want to repeat experiments, set the N_REPEAT variable in config.py

# understand the array keyword before using it
# --array=0%1 executes job 0 in one process
# --array=0-2%1 executes job 0,1,2 in one process after another
# --array=0-2%3 executes job 0,1,2 in three processes simultaneously

# change the resources flags if needed:
# try to save resource, do not use more RAM than needed, try not
# to block free GPUs, we need all the capacity! 
 