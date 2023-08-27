#This file takes a generated model and uses it to sort the given files.
#We assume you are using two tab seperated documents. If not, follow the instructions
#in trainingmodel.py. As stated, there are no arguments, and you just run this python
#file. Any settings must be changed and saved here. Make sure all your libraries are
#installed.

import pandas as pd
import time
from splink.duckdb.linker import DuckDBLinker
import splink.duckdb.comparison_library as cl
import splink.duckdb.comparison_template_library as ctl
import splink.duckdb.blocking_rule_library as brl
from splink.datasets import splink_datasets
import numpy as np
from sklearn.datasets import load_iris
from IPython.display import display
import duckdb

start_time = time.time()

#replace ds7.1.1 and ds7.1.2 with your tab seperated input file paths. If the format is
#different, replace the sep='\t' with whatever character is your seperator 

df1 = pd.read_csv("ds7.1.1", sep='\t', names=['unique_id','last_name','first_name','DOB','DOD'] )

df2 = pd.read_csv("ds7.1.2", sep='\t', names=['unique_id','last_name','first_name','DOB','DOD'])

result = pd.concat([df1, df2])

print(result)



df = result

linker = DuckDBLinker(df)
#model.json is the pretrained model. You can use given models from their library or train one
#following the instructions in trainingmodel.py.
linker.load_model("model.json")
pairwise_predictions = linker.predict()


clusters = linker.cluster_pairwise_predictions_at_threshold(pairwise_predictions, 0.95)
clusters.as_pandas_dataframe(limit=5)

print("--- %s seconds ---" % (time.time() - start_time))
#This prints the run-time.

#saves the model to a file

clusters.to_csv('clusters2.csv')

sql = f"select * from {clusters.physical_name} limit 2 "
dfc = linker.query_sql(sql)
print(dfc)

linker.precision_recall_chart_from_labels_column("unique_id", threshold_actual = 0.95)