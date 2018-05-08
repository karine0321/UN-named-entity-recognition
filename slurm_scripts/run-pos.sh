#!/bin/sh
#
#SBATCH --verbose
#SBATCH --job-name=NER-pos-testdata
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --mem=60GB
#SBATCH --cpus-per-task=14
#SBATCH --mail-type=END
#SBATCH --mail-user=lh1036@nyu.edu

module purge
module load python3/intel/3.6.3
RUNDIR=$SCRATCH/my_project/run-${SLURM_JOB_ID/.*}
mkdir -p $RUNDIR

echo $RUNDIR

cp main_pos.py $RUNDIR
cp sentences.py $RUNDIR
cp taggers.py $RUNDIR
cp tagged-test -r $RUNDIR
cp -r ner_env/ $RUNDIR

export RUNDIR

cd $RUNDIR

ls

source ner_env/bin/activate

python3 main_pos.py tagged-test temp_features

exit
