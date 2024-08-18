# <div align="center">Multimodal-Sentiment-and-Stance-Detection</div>

## <div align="left">Utilisation Guide 🚀</div>

1- Download the Singularity image from this link and copy it into the project folder

2- Add videos that you want to make analysis on to the Videos folder

3- Run the slurm file with commind line 
```bash 
sbacth template_segmentation.slurm "Gun Control"
```
4- After some time you will find a new file added that start with "Biase_Results" which will have the result of biase anaysis of the indivusal speaker and the global speaker and the host (if founed)

