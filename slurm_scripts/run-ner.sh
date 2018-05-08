#!/bin/sh
#
#SBATCH --verbose
#SBATCH --job-name=NER-classifier
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH --time=24:00:00
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

cp main_ner.py $RUNDIR
cp sentences.py $RUNDIR
cp taggers.py $RUNDIR
cp preppedNERSentences-training -r $RUNDIR
cp preppedNERSentences-test -r $RUNDIR
cp -r ner_env/ $RUNDIR

export RUNDIR

cd $RUNDIR

ls

source ner_env/bin/activate

python3 main_ner.py preppedNERSentences-training/ preppedNERSentences-test/ 200 ner-output/ classifier_out.json test_out.json

exit
