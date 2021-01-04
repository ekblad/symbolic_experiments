# symbolic_experiments  
[![DOI](https://zenodo.org/badge/234428461.svg)](https://zenodo.org/badge/latestdoi/234428461)   
Repository for symbolic regression/classification experiments. The experiment files are `sc.py` and `sr.py`, run on a SLURM cluster using `sbatch sr.sh`. The number of trials are specified using a job array index in the shell scripts, and SLURM will schedule these sequentially. Other scripts within the experiment directories are provided to collect and analyze model results.

## Running locally -

Download data set from here: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4360593.svg)](https://doi.org/10.5281/zenodo.4360593)

Code will look for this file (`'sym_data_4.csv'`) alongside repository (in same directory that `symbolic_experiments` is in).

Run the following line from the same directory as `sc.py` or `sr.py`, and scoop will automatically collect available processors to distribute model evaluations over. You can watch this happen in your system's task manager.  

`python -m scoop sc.py 1`  

The integer provided as an argument at the end (the job index in the cluster version) will be used to initialize random processes, do training-test splits, and label the trial folder and results files.

## Dependencies and versions:
### For experiments:

Required:  
`python                    3.7.1`  
`numpy                     1.16.1`  
`pandas                    0.25.3`   
`scipy                     1.2.0`  
`deap                     1.2.2`  
`scoop                     0.7.1.1`(as written, can be modified to run single processor)  

Optional*(for immediate plotting of results):  
`matplotlib                3.1.0`  
`pygraphviz                1.3`  
`scikit-learn              0.20.2` 

*Remote installations of python may not have plotting, so can turn these off in `sr.py` and `sc.py`. All results are stored in a pickle within a directory labeled with the trial number.

 ### For sensitivity testing, clustering, and most figures:  
`salib                     1.3.8`  
`scikit-learn              0.20.2`  
`matplotlib                3.1.0`  
`seaborn                   0.9.0`  

### For networks and maps:  
`pygraphviz                1.3`  
`cartopy                   0.17.0`  

## Other data:

The code that extracts the IWFM water pumping and delivery estimates from C2VSim can be found [here](https://github.com/giorgk/C2Vsim_FG_v2/tree/master/NSF_Liam).  

The code that extracts the land use data from the California Pesticide Use Reports can be found [here](https://github.com/nataliemall/crop_acreages_from_DPR_reports).   

