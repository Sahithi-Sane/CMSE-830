import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np

cancer = pd.read_csv("data.csv") # load data
cancer.drop(["Unnamed: 32"], axis=1, inplace=True) #drop unnamed/NaN column

st.text("Let's explore a sample of the dataset") # prints in web app
st.dataframe(cancer.head()) # prints head in web app

cols = st.multiselect(
    "Please choose columns to generate pairplot", list(cancer.columns[2:,]), ["radius_mean", "texture_mean"]) # generates multiselect window with some defaults, excludes id and diagnosis

if len(cols) < 2:
    st.error("Please select at least two columns.") # tags an error when less than two columns are selected
else:
    cols.insert(0, 'diagnosis') # inserts diagnostic label back in
    pair = sns.pairplot(cancer[cols], hue = "diagnosis", palette="flare") # pairplot generation
    st.text("Pairplot of Values with Diagnostic Hue") # title text
    st.pyplot(pair.fig) #prints to web app



import matplotlib.pyplot as plt # only using matplotlib to generate axes, not create plots

sns.set_theme(style="white")
corr = cancer.iloc[:, 2:len(cancer.columns)].corr()
# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))
# Set up the matplotlib figure
fig, ax = plt.subplots(figsize=(11, 9))
sns.heatmap(corr, mask=mask, square=True, linewidths=.5, cbar_kws={"shrink": .6}, ax=ax)

st.text("Correlation heatmap of all variables") #write figure title
st.write(fig) # print fig in web app


st.text("Pie Chart of Diagnostics") #write figure title

fig, ax = plt.subplots(figsize=(11, 9))
palette_color = sns.cubehelix_palette(2)
data = [cancer['diagnosis'].value_counts()['B'], cancer['diagnosis'].value_counts()['M']]
ax.pie(data, labels=['Benign: '+str(data[0]), 'Malignant: '+str(data[1])], colors=palette_color, autopct='%.0f%%', shadow=True, explode=[0.1,0])
st.pyplot(fig)


st.text("Distribution Plot")

# displot = sns.displot(cancer.iloc[:, [2,3,4]],kind = 'kde',rug = 'True', color="blue")
# st.pyplot(displot.fig)


fig, ax = plt.subplots(figsize=(11, 9))
sns.distplot(cancer.iloc[:, 2], color='lightpink', label="radius mean", ax=ax)
sns.distplot(cancer.iloc[:, 3], color='mediumorchid', label="texture mean", ax=ax)
sns.distplot(cancer.iloc[:, 4], color='darkslateblue', label="perimeter mean", ax=ax)
ax.set_xlabel("Distribution")
ax.set_ylabel("Density")
ax.legend()
st.pyplot(fig)