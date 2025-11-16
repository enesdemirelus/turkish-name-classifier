import pandas as pd

m = pd.read_csv("male_name_tally.csv", usecols=[0], header=None, names=["name"])
f = pd.read_csv("female_name_tally.csv", usecols=[0], header=None, names=["name"])
u = pd.read_csv("unisex_name_tally_filtered.csv", usecols=[0], header=None, names=["name"])

all_names = pd.concat([m,f,u]).drop_duplicates().reset_index(drop=True)

all_names.to_csv("names.csv", index=False)
