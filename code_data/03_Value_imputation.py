import argparse
import pandas as pd
import numpy as np


data= pd.read_csv("")
# carrying over the values to a new column called lems
df['lems_q6'] = df['c_lems'].fillna(df["aiii_lems"])
df['lems_q6_q3'] = df['lems_q6'].fillna(df["aii_lems"])

# carrying over the asia score to a new column called asia_chronic
#"va_ais", "ai_ais", "aii_aiis", "aiii_aiiis", "c_cs"
df['asia_chronic_q6'] = df['c_cs'].fillna(df["aiii_aiiis"])
df['asia_chronic_q6_q3'] = df['asia_chronic_q6'].fillna(df["aii_aiis"])


# index for patients who have all 0 values
all_0_index=[] # insert 0

df["va_lems_imputed"]=df["va_lems"]
for i in range(len(all_0_index)):
    # create a new columns with the imputed values
    df.at[all_0_index[i],"va_lems_imputed"]= 0

# removed due to abnormal decrease in LEMS score
df=df.drop(index=[])

df.to_csv("")