import os
import pandas as pd

dir_path = os.path.dirname(os.path.realpath(__file__))
local_path = os.path.join(dir_path,'model_sensitivities.csv')

data = pd.read_csv(local_path,header=0,index_col=0,low_memory=False)
data = data[data.experiment=='classification']
for i in data.columns:
	print(i)

# exit()
model_list = ['seed_2_model_54','seed_3_model_51','seed_6_model_33']

for model in model_list:
	print(data[data.model_id==model].labels.unique())