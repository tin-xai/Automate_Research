# %%
import numpy as np
import pandas as pd
import math 
import requests
import time
from tqdm.notebook import tqdm
tqdm.pandas()

from pandarallel import pandarallel

pandarallel.initialize(progress_bar=True)

def function_to_apply(i):
    sum = 0
    for i in range(10000000):
        sum +=i
    return 1

df = pd.read_csv("cub_df.csv")
df.head(10)

# %%
t1 = time.time()
df["Site"] = df["Site"].parallel_apply(function_to_apply)

time.time()-t1
# %%

t2 = time.time()
df["Site"] = df["Site"].progress_apply(function_to_apply)
time.time()-t2
# %%
