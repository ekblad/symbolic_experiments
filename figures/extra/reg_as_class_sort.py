import os
import pandas as pd
import pareto

dir_path = os.path.dirname(os.path.realpath(__file__))
# local_path = os.path.join(dir_path,'model_reg_as_class.csv')
local_path = os.path.join(dir_path,'model_class_acc.csv')

data = pd.read_csv(local_path,header=0,index_col=0,low_memory=False)

of_cols = [1,3,5] # sorting by test error

data_nondom = pareto.eps_sort([list(data.itertuples(False))], of_cols)
data_sorted = pd.DataFrame.from_records(data_nondom, columns=list(data.columns.values))
data_sorted['sum'] = data_sorted.sum(axis=1)

print(data_sorted.min())