#!/bin/bash


#SBATCH --partition=normal
#SBATCH --cpus-per-task=10
# memory in MB
#SBATCH --mem=20G
# The %04a is translated into a 4-digit number that encodes the SLURM_ARRAY_TASK_ID
#SBATCH --output=results_sequential/image_translation_exp_%04a_stdout.txt
#SBATCH --error=results_sequential/image_translation_exp_%04a_stderr.txt
#SBATCH --time=48:00:00
#SBATCH --job-name=Homework_08
#SBATCH --mail-user=harinadhappidi@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/cs504302/Homework_08/
#SBATCH --array=0-4

#################################################
# Do not change this line unless you have your own python/tensorflow/keras set up
. /home/fagg/tf_setup.sh
conda activate tf

# Change this line to start an instance of your experiment

python hw8_base.py @Sequential_UNET.txt --fold $SLURM_ARRAY_TASK_ID

#python hw8_base.py @Model_UNET.txt --fold $SLURM_ARRAY_TASK_ID
 