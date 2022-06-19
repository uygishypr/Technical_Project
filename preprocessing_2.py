#%%
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

#%%
df_energy_and_materials = pd.read_excel("sector_data/Energy_and_Materials.xlsx")
df_consumer_discretionary = pd.read_excel("sector_data/Consumer_Discretionary.xlsx")
df_consumer_staples = pd.read_excel("sector_data/Consumer_Staples.xlsx")
df_industrials = pd.read_excel("sector_data/Industrials.xlsx")
df_energy_and_materials.head()

#%%
frames = [df_energy_and_materials,df_consumer_discretionary,
          df_consumer_staples, df_industrials]

df_combined = pd.concat(frames)

#%%
final_df_columns = ["".join(name.split()[:-1]) for name in df_combined.filter(regex="FY0$", axis=1)]
df_merged = pd.DataFrame(columns=["Company Name"] + ["GICS Sector Name"] + ["Identifier (RIC)"]+final_df_columns)

for i, row in df_combined.iterrows():
    for j in range(10):
        data = pd.Series(data=row.filter(regex=f"FY{j}$").values, index=final_df_columns)
        data["Company Name"] = row["Company Name"]
        data["GICS Sector Name"] = row["GICS Sector Name"]
        data["Identifier (RIC)"] = row["Identifier (RIC)"]
        df_merged = df_merged.append(data, ignore_index=True)
    print("iteration: ", i)

# %%
df_merged.to_excel("Series_dataset_2.xlsx")

#%%
#%%
df_merged.drop("ESGScore", axis=1, inplace = True)
# %%
df_merged.head()
rule = lambda x: 1 if x>=1 else 0
controversy_cols = df_merged.iloc[:, 3:26]
controversy_cols.sum(axis=1)
df_merged["Label"] = controversy_cols.sum(axis=1).apply(rule)

#%%
# Since the categories TRUE and FALSE are considered boolean variables, they can be summed as 1 and 0's
type(df_merged["ResponsibleMarketingControversies"][0])

#%%

# The imbalance between classes can be seen from here 
df_merged["Label"].value_counts()

df_merged.drop(controversy_cols.columns.values.tolist(), axis=1,inplace=True)

df_merged.head()

#%%

# High percentages mean more missing values in proportion
print(df_merged.isna().mean())  

# Identify the columns that have more than half of their values missing
df_merged[df_merged.columns[df_merged.isnull().mean() > 0.5]]

#%%

df_merged.set_index(df_merged["Company Name"],drop=True, inplace = True)

#%%
df_merged.head()




# #%%
# # Extract the controversies
# controversies_section = df_combined.iloc[:,13:243] 
# controversies_section.head(10)
# #%% 
# # Extract the features
# features_section = df_combined.iloc[:,244:]
# features_section.head(10)

# #%%
# # Group the controversies in a specific year
# fy0 = controversies_section.filter(regex="FY0$", axis=1)
# fy1 = controversies_section.filter(regex="FY1$", axis=1)
# fy2 = controversies_section.filter(regex="FY2$", axis=1)
# fy3 = controversies_section.filter(regex="FY3$", axis=1)
# fy4 = controversies_section.filter(regex="FY4$", axis=1)
# fy5 = controversies_section.filter(regex="FY5$", axis=1)
# fy6 = controversies_section.filter(regex="FY6$", axis=1)
# fy7 = controversies_section.filter(regex="FY7$", axis=1)
# fy8 = controversies_section.filter(regex="FY8$", axis=1)
# fy9 = controversies_section.filter(regex="FY9$", axis=1)


# fy0.insert(0, "Company Name",df_combined["Company Name"].tolist(), True)
# fy0.head()

# # %%

