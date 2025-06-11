#!/usr/bin/env python3
import csv


import pandas as pd
import glob

# Step 1: Find all CSV files in the folder
#csv_files = glob.glob("*.csv")  # or "path/to/folder/*.csv"




csv_files = ['DataBase_2.csv', 'DataBase_3.csv', 'DataBase_4.csv']#, 'DataBase_4.csv']
#csv_files = ['DataBase_2.csv', 'DataBase_3.csv', 'DataBase_1.csv', 'DataBase_4.csv', 'Berlisov_monograph_Sarmatian_burials.csv']


print("files:  ", csv_files)

# Step 2: Load and concatenate
df_list = [pd.read_csv(file, header=None) for file in csv_files]
merged_df = pd.concat(df_list, ignore_index=True)

# Step 3: Save merged file
merged_df.to_csv("DataSets234.csv", index=False, header=False)

print(f"Merged {len(csv_files)} files into merged_output.csv")
