#This file is used to train a model. Assuming you have two datasets to concatenate and 
#then use to train a model, input their paths below. Make sure you have the modules below
#installed, I have not yet made a requirements.txt so just use pip.

#To run, just use python3 to run the file. There are no arguments, you just change the
#settings in this file

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

#replace ds1.1.1 and ds1.1.2 with your tab seperated input file paths. If the format is
#different, replace the sep='\t' with whatever character is your seperator 

df1 = pd.read_csv("ds1.1.1", sep='\t', names=['unique_id','last_name','first_name','DOB','DOD'] )

df2 = pd.read_csv("ds1.1.2", sep='\t', names=['unique_id','last_name','first_name','DOB','DOD'])

result = pd.concat([df1, df2])

print(result)



df = result

#assumung your file format matches the ones we've used, don't touch the settings below


settings = {
    "link_type": "dedupe_only",
    "blocking_rules_to_generate_predictions": [
        "l.last_name = r.last_name",
    ],
    "comparisons": [
        ctl.name_comparison("first_name"),
        ctl.name_comparison("last_name"),
        ctl.name_comparison("DOB"),
    ],
}

linker = DuckDBLinker(df, settings)
linker.estimate_u_using_random_sampling(max_pairs=1e6)

blocking_rule_for_training = brl.and_(
                                brl.exact_match_rule("last_name")
                                )



linker.estimate_parameters_using_expectation_maximisation(blocking_rule_for_training)

blocking_rule_for_training = brl.exact_match_rule("DOB")
linker.estimate_parameters_using_expectation_maximisation(blocking_rule_for_training)

pairwise_predictions = linker.predict()



clusters = linker.cluster_pairwise_predictions_at_threshold(pairwise_predictions, 0.95)
clusters.as_pandas_dataframe(limit=5)

print("--- %s seconds ---" % (time.time() - start_time))
#This prints the run-time.

#saves the model to a file
linker.save_model_to_json("model.json")

clusters.to_csv('clusters.csv')

