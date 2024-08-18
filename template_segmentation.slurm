#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 2
#SBATCH -p gpu --gres=gpu:1
#SBATCH -C gpul40s
#SBATCH --mem-per-gpu=32G
#SBATCH --time=0-02:00:00       
#SBATCH --job-name="mak417_test_llama"
#SBATCH --output=/scratch/users/mak417/test.log

nvidia-smi

cd /scratch/users/mak417

# Load Singularity module

module load singularity

# If Singularity image does not exist, pull image and copy to safe



# Copy code from safe
scp hpc4://home/mak417/mak417gallinahome/safe/my_image.sif ./
scp hpc4://home/mak417/mak417gallinahome/safe/video_segmentation.py ./
scp hpc4://home/mak417/mak417gallinahome/safe/Llama_3_8B.py ./
scp -r hpc4://home/mak417/mak417gallinahome/safe/Videos ./Videos


# Run experiments:
echo "Subject: $1"


for f in Videos/*; do
    # Print the file name
    echo "Processing file: $f"
    
    # Copy the file using scp
    
    # Run the video segmentation script
    singularity exec --nv -H /scratch/users/mak417/ /scratch/users/mak417/my_image.sif python3 video_segmentation.py "$1" "$f"
    
    # Run the Llama_3_8B script
    singularity exec --nv -H /scratch/users/mak417/ /scratch/users/mak417/my_image.sif python3 Llama_3_8B.py  "$1" "$f"
    
    # Remove the original video file
    rm "orginal_video.mp4"
    rm -r speaker_parts_audio
    rm -r speaker_parts

done

subject=$1
filename="Biase_Results_${subject}.txt"
scp $filename hpc4://home/mak417/mak417gallinahome/safe/$filename


scp test.log hpc4://home/mak417/mak417gallinahome/safe/test.log

# Remove files
# rm -r mdd_experiments
# rm -r l2-arctic
# rm l2-arctic.zip
# rm image.sif