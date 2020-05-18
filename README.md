# symbolic_experiments
Repository for symbolic regression/classification experiments.

## Running locally -

Download data set from here:
Code will look for this file (`'sym_data_4.csv'`) alongside repository (into same directory where you downloaded symbolic_experiments)

Run the following line from the same directory as `sc.py` or `sr.py`, and scoop will automatically collect available processors to distribute model evaluations over. You can watch this happen in your system's task manager.

## Dependencies and versions:

`deap                     1.2.2`
`matplotlib                3.1.0`
`numpy                     1.16.1`
`pandas                    0.25.3`
`pygraphviz                1.3`
`python                    3.7.1`
`salib                     1.3.8`
`scikit-learn              0.20.2`
`scipy                     1.2.0`
`scoop                     0.7.1.1`
`seaborn                   0.9.0`

Remote installations of python may not have plotting, so can turn these off in `sr.py` and `sc.py`. All results are stored in a pickle within a directory labeled with the trial number.
