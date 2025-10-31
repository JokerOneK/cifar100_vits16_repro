import pandas as pd

df = pd.read_csv('./final_results_no_checkpoint_2_epochs/layer_times.csv.gz', compression='gzip')

print(df.head(24))