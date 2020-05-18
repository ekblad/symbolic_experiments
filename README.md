# symbolic_experiments
Repository for symbolic regression/classification experiments.

## Running locally -

Download data set from here:  

Code will look for this file (`'sym_data_4.csv'`) alongside repository (into same directory where you downloaded symbolic_experiments)

Run the following line from the same directory as `sc.py` or `sr.py`, and scoop will automatically collect available processors to distribute model evaluations over. You can watch this happen in your system's task manager.  

`python -m scoop sc.py [1]`  

The integer provided as an argument will be used to initialize random processes and label the trial.

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


