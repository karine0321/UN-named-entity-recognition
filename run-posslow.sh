#!/bin/sh
#
#SBATCH --verbose
#SBATCH --job-name=NER-pos-singletest
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --mem=60GB
#SBATCH --cpus-per-task=6
#SBATCH --mail-type=END
#SBATCH --mail-user=lh1036@nyu.edu

module purge
module load python3/intel/3.6.3
RUNDIR=$SCRATCH/my_project/run-${SLURM_JOB_ID/.*}
mkdir -p $RUNDIR

echo $RUNDIR

cp main.py $RUNDIR
cp rawdata.py $RUNDIR
cp tagged-training -r $RUNDIR
cp -r ner_env/ $RUNDIR

export RUNDIR

cd $RUNDIR

ls

source ner_env/bin/activate

python3 main.py tagged-training tagged-test temp_features

exit
