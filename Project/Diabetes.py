import streamlit as st
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import mean 
from numpy import std
import os
import warnings

st.title('Diabetis Dataset')
st.caption('Presented by Sahithi Sane')
st.divider()

# Load the dataset into the dataframe
df_train = pd.read_csv('https://raw.githubusercontent.com/Sahithi-Sane/CMSE-830/main/Project/TrainingWiDS2021.csv')

#Collecting all the columns with apache score
all_data = df_train
apache_cols = [col for col in all_data.columns if '_apache' in col]


apache_cols = [c.split('_apache')[0] for c in apache_cols]
print(apache_cols)

vital_cols = all_data.columns[all_data.columns.str.startswith('d1') & all_data.columns.str.contains('_max')]
vital_cols = [(c.split('d1_')[1]).split('_max')[0] for c in vital_cols]

common_cols = [c for c in apache_cols if c in vital_cols]

for c in common_cols:
    var1 = f"d1_{c}_max"
    var2 = f"{c}_apache"
    notna_condition = all_data[var1].notna() & all_data[var2].notna()

#print(f"{c} has {np.round((all_data[notna_condition][var2] == (all_data[notna_condition][var1])).sum()/len(all_data[notna_condition])*100,2)}% duplicates with {var1}")
st.write("Fro Apache score displaying which columns have highest similarity :")
percentage_duplicates = np.round((all_data[notna_condition][var2] == all_data[notna_condition][var1]).sum() / len(all_data[notna_condition]) * 100, 2)

# Display the information using Streamlit
st.write(f"{c} has {percentage_duplicates}% duplicates with {var1}")


# Check if there are any missing values in the dataset and drop them
all_data["d1_heartrate_max"] : np.where((all_data["d1_heartrate_max"].isna()
                                      & all_data["heart_rate_apache"].notna()),
                                     all_data["heart_rate_apache"],
                                     all_data["d1_heartrate_max"])

all_data.drop("heart_rate_apache", axis=1, inplace=True)

cols_to_drop = []
for i, col_1 in enumerate(all_data.columns):
     for col_2 in all_data.columns[(i+1):]:
            if all_data[col_1].equals(all_data[col_2]):
                print(f"{col_1} and {col_2} are identical.")
                cols_to_drop.append(col_2)
                
all_data.drop(cols_to_drop, axis=1, inplace = True)

all_data["d1_pao2fio2ratio_max"] = np.where((all_data["pao2_apache"].notna()
                                           & all_data["fio2_apache"].notna()
                                           & all_data["d1_pao2fio2ratio_max"].isna() ),
                                           all_data["pao2_apache"] / all_data["fio2_apache"],
                                           all_data["d1_pao2fio2ratio_max"])

drop_columns = all_data.columns[all_data.columns.str.startswith('h1')]
all_data.drop(drop_columns, axis=1, inplace=True)

all_data = all_data[all_data['age'] >= 16].reset_index(drop=True)
all_data['ethnicity'] = all_data['ethnicity'].fillna('Other/Unknown')
all_data['gender'] = all_data['gender'].fillna(all_data['gender'].mode())[0]

all_data["weight"] = np.where((all_data["weight"].isna()
                              & all_data["bmi"].notna()),
                              all_data["bmi"],
                              all_data["weight"])

all_data["height"] = np.where((all_data["height"].isna()
                              & all_data["weight"].notna()),
                             all_data["weight"],
                             all_data["height"])

all_data['height'] = all_data.groupby('gender')['height'].transform(lambda x: x.fillna(x.mean()))
all_data['weight'] = all_data.groupby('gender')['weight'].transform(lambda x: x.fillna(x.mean()))
all_data['bmi'] = all_data.groupby('gender')['bmi'].transform(lambda x: x.fillna(x.mean()))

all_data.loc[all_data['hospital_admit_source'] == 'Acute Care/Floor', 'hospital_admit_source'] = 'Floor'
all_data.loc[all_data['hospital_admit_source'] == 'Step-Down Unit (SDU)', 'hospital_admit_source'] = 'SDU'
all_data.loc[all_data['hospital_admit_source'] == 'ICU to SDU', 'hospital_admit_source'] = 'SDU'
all_data.loc[all_data['hospital_admit_source'] == 'Other ICU', 'hospital_admit_source'] = 'ICU'
all_data.loc[all_data['hospital_admit_source'] == 'PACU', 'hospital_admit_source'] = 'Recovery Rom'
all_data.loc[all_data['hospital_admit_source'] == 'SDU', 'hospital_admit_source'] = 'ICU'
all_data.loc[all_data['hospital_admit_source'] == 'Chest Pain Center', 'hospital_admit_source'] = 'Other'
all_data.loc[all_data['hospital_admit_source'] == 'Observation', 'hospital_admit_source'] = 'Other'

all_data['hospital_admit_source'].fillna('Other', inplace = True)
all_data['icu_admit_source'].fillna(all_data['hospital_admit_source'], inplace = True)
 
all_data.loc[all_data['icu_admit_source'] == 'Operating Room', 'icu_admit_source'] = 'Operating Room / Recovery'
all_data.loc[all_data['icu_admit_source'] == 'Emrgency Department', 'icu_admit_source'] = 'Accident & Emergency'
all_data.loc[all_data['icu_admit_source'] == 'Direct Admit', 'icu_admit_source'] = 'Other'



st.title("Select desired X and Y Variables from the Heart Disease Dataset")
x_variable = st.selectbox("X Variable", all_data.columns)
y_variable = st.selectbox("Y Variable", all_data.columns)

data_button = st.selectbox('Please choose preferred visualization', ['Scatter Plot', 'Heatmap', 'Histogram Plot', 'Line Plot', 'Boxplot', 'Relational Plot', 'Distribution Plot'])

if data_button == 'Scatter Plot':
    scatter_plot = sns.scatterplot(data=all_data, x=x_variable, y=y_variable)
    st.pyplot(scatter_plot.figure)

elif data_button == 'Heatmap':
    plt.figure(figsize=(10,10))
    heatmap = sns.heatmap(all_data.corr(numeric_only=True), annot=True, cmap="crest")
    st.pyplot(heatmap.figure)

elif data_button == 'Histogram Plot':
    histplot = sns.histplot(data=all_data, x=x_variable, binwidth=3)
    st.pyplot(histplot.figure)

elif data_button == 'Line Plot':
    lineplot = sns.lmplot(x=x_variable, y=y_variable, hue="diabetes_mellitus", data=all_data)
    st.pyplot(lineplot.figure)

elif data_button == 'Boxplot':
    boxplot = sns.boxplot(x=x_variable, hue='diabetes_mellitus', data = all_data)
    st.pyplot(boxplot.figure)

elif data_button == 'Relational Plot':
    relplot = sns.relplot(all_data, x=x_variable, y=y_variable, hue="diabetes_mellitus", kind="line")
    st.pyplot(relplot.figure)

elif data_button == 'Distribution Plot':
    distplot = sns.displot(all_data, x=x_variable, hue = "diabetes_mellitus", col="gender", kind="kde", rug=True)
    st.pyplot(distplot.figure)
    