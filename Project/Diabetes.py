import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.offline as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.tools as tls
import plotly.figure_factory as ff
py.init_notebook_mode(connected=True)
import squarify
import altair as alt
import plotly.express as px
import streamlit as st
import hiplot as hip

#------------------------------------------------------------------------------------------------------------------------------------------
st.title('Diabetis Dataset')
st.caption('Presented by Sahithi Sane')
st.divider()

# Load the dataset into the dataframe
df_data = pd.read_csv('https://raw.githubusercontent.com/Sahithi-Sane/CMSE-830/main/Project/diabetes.csv')
df_temp = df_data

D = df_data[(df_data['Outcome'] != 0)]
H = df_data[(df_data['Outcome'] == 0)]

def my_function():
    st.caption("Heart Disease Data Visualization")
    selected_variable = st.selectbox("Select the desired variable", df_data.columns)
    scatter_plot = alt.Chart(df_data).mark_circle(size=60, opacity=0.7).encode(
        x=selected_variable,
        y='Outcome:N',
        tooltip=[selected_variable, 'Outcome']
    ).properties(
        width=600,
        height=400
    )
    st.write(scatter_plot)

# Missing Value analysis functions
def missing_plot(dataset, key):
    null_feat = pd.DataFrame(len(dataset[key]) - dataset.isnull().sum(), columns=['Count'])
    percentage_null = pd.DataFrame((len(dataset[key]) - (len(dataset[key]) - dataset.isnull().sum())) / len(dataset[key]) * 100, columns=['Count'])
    percentage_null = percentage_null.round(2)

    trace = go.Bar(x=null_feat.index, y=null_feat['Count'], opacity=0.8, text=percentage_null['Count'], textposition='auto', marker=dict(color='#7EC0EE',
        line=dict(color='#000000', width=1.5)))

    layout = dict(title="Missing Values of total records i.e. " + str(len(dataset)) + " (count & Percentage)")
    fig = dict(data=[trace], layout=layout)
    st.plotly_chart(fig)

#To replace missing values, we'll use median by target (Outcome)
def median_target(var):   
    temp = df_data[df_data[var].notnull()]
    temp = temp[[var, 'Outcome']].groupby(['Outcome'])[[var]].median().reset_index()
    return temp

# Function to show and perform Missing Value Analysis
def missing_value_analysis():
    df_data[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = df_data[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)
    missing_plot(df_data, 'Outcome')
    st.write("""
Missing values :

- Insulin = 48.7% - 374
- SkinThickness = 29.56% - 227
- BloodPressure = 4.56% - 35
- BMI = 1.43% - 11
- Glucose = 0.65% - 5
            """)
    st.write("To replace missing values, we'll use median by target (Outcome)")
    df_data.loc[(df_data['Outcome'] == 0 ) & (df_data['Glucose'].isnull()), 'Glucose'] = 107.0
    df_data.loc[(df_data['Outcome'] == 1 ) & (df_data['Glucose'].isnull()), 'Glucose'] = 140.0
    median_target('BloodPressure')
    df_data.loc[(df_data['Outcome'] == 0 ) & (df_data['BloodPressure'].isnull()), 'BloodPressure'] = 70
    df_data.loc[(df_data['Outcome'] == 1 ) & (df_data['BloodPressure'].isnull()), 'BloodPressure'] = 74.5
    median_target('SkinThickness')
    df_data.loc[(df_data['Outcome'] == 0 ) & (df_data['SkinThickness'].isnull()), 'SkinThickness'] = 27.0
    df_data.loc[(df_data['Outcome'] == 1 ) & (df_data['SkinThickness'].isnull()), 'SkinThickness'] = 32.0
    median_target('Insulin')
    df_data.loc[(df_data['Outcome'] == 0 ) & (df_data['Insulin'].isnull()), 'Insulin'] = 102.5
    df_data.loc[(df_data['Outcome'] == 1 ) & (df_data['Insulin'].isnull()), 'Insulin'] = 169.5
    median_target('BMI')
    df_data.loc[(df_data['Outcome'] == 0 ) & (df_data['BMI'].isnull()), 'BMI'] = 30.1
    df_data.loc[(df_data['Outcome'] == 1 ) & (df_data['BMI'].isnull()), 'BMI'] = 34.3
    st.write("After replacing the missing values by meadian by target we can see that the missing value handeling is done.")
    missing_plot(df_data, 'Outcome')

# Function to generate a Histogram
def create_histogram(df_data, x_column, title):
    histogram = px.histogram(df_data, x=x_column, nbins=100, marginal="rug", color_discrete_sequence=["#FF6F61"])
    histogram.update_layout(
        title=title,
        xaxis_title=x_column,
        yaxis_title="Frequency",
    )
    st.plotly_chart(histogram)

# Function to generate a Boxplot
def create_boxplot(df_data, x_column,title):
    boxplot = px.box(df_data, x=x_column, color_discrete_sequence=["#6A0572"])
    boxplot.update_layout(
        title=title,
        xaxis_title="",
        yaxis_title=x_column,
    )
    st.plotly_chart(boxplot)

# Function to generate a Violin plot
def create_violin_plot(df_data, x_column, title):
    violin_plot = px.violin(df_data, x=x_column, box=True, color_discrete_sequence=["#00A896"])
    violin_plot.update_layout(
        title=title,
        xaxis_title="",
        yaxis_title=x_column,
    )
    st.plotly_chart(violin_plot)

# Function to plot Distribution density plot
def plot_distribution(data_select, size_bin) :  
    # 2 datasets
    tmp1 = D[data_select]
    tmp2 = H[data_select]
    hist_data = [tmp1, tmp2]

    group_labels = ['diabetic', 'healthy']
    colors = ['#FFD700', '#7EC0EE']
    fig = ff.create_distplot(hist_data, group_labels, colors=colors, show_hist=True, bin_size=size_bin, curve_type='kde')
    fig['layout'].update(title= "Distribution Plot" + " - " + data_select)
    
    return fig

# Univariate Exploratory Data Analysis
def uni_eda():
    data_button = st.selectbox('Please choose preferred visualization', ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'Age', 'Outcome', 'Diabetes Pedigree Function'])
    st.write("There are many plots to analyse these type of data. Histograms, Box plots and Violin plots, are useful to know how the data is distributed for quantitative features.")
    if data_button == 'Pregnancies':
        histogram = create_histogram(df_data, x_column="Pregnancies", title="Pregnancies Distribution (Histogram)")
        boxplot = create_boxplot(df_data, x_column="Pregnancies", title="Pregnancies Distribution (Boxplot)")
        violin_plot = create_violin_plot(df_data, x_column="Pregnancies", title="Pregnancies Distribution (Violin Plot)")
        st.plotly_chart(plot_distribution('Pregnancies', 0))
        st.markdown('''From the above analysis we observe that:
-> Most patients had 0, 1 or 2 pregnancies.       
-> Median value of Pregnancies is 3.        
-> Individuals with diabetes tend to have more or fewer pregnancies        
-> Also, patients had upto 17 pregnancies!                   
There are 3 outliers on the boxplot which we can neglect.
                    
                    ''')

    elif data_button == 'Glucose':
        histogram = create_histogram(df_data, x_column="Glucose", title="Glucose Distribution (Histogram)")
        boxplot = create_boxplot(df_data, x_column="Glucose", title="Glucose Distribution (Boxplot)")
        violin_plot = create_violin_plot(df_data, x_column="Glucose", title="Glucose  Distribution (Violin Plot)")
        st.plotly_chart(plot_distribution('Glucose', 0))
        st.markdown('''We observe that:              
-> Median (117.0) and mean (120.8) of Glucose lie very close to each other i.e. the distribution is more or less symmetric and uniform.       
-> As seen from the box plot, an outlier lies on 0-value             
-> Individuals with diabetes tend to have higher or lower glucose levels on average                   
-> The boxplot has a significant difference in medians between diabetic and non-diabetic groups, it can suggest that the "Glucose" feature might be associated with diabetes
                    
                    ''')

    elif data_button == 'BloodPressure':
        histogram = create_histogram(df_data, x_column="BloodPressure", title="Blood Pressure Distribution (Histogram)")
        boxplot = create_boxplot(df_data, x_column="BloodPressure", title="Blood Pressure Distribution (Boxplot)")
        violin_plot = create_violin_plot(df_data, x_column="BloodPressure", title="Blood Pressure Distribution (Violin Plot)")
        st.plotly_chart(plot_distribution('BloodPressure', 4))        
        st.markdown('''We observe that:             
-> Median (72.0) and mean (69.1) of BloodPressure lie very close to each other i.e. the distribution is more or less symmetric and uniform.            
-> As seen from the box plot and violin plot, some outliers lie on 0-value.           
-> histogram showS that individuals with diabetes tend to have higher blood, which can be a potential risk factor              
-> The boxplot reveal that the median blood pressure for diabetics is significantly higher than for non-diabetics.
                 
                     ''')

    elif data_button == 'SkinThickness':
        histogram = create_histogram(df_data, x_column="SkinThickness", title="SkinThickness Distribution (Histogram)")
        boxplot = create_boxplot(df_data, x_column="SkinThickness", title="SkinThickness Distribution (Boxplot)")
        violin_plot = create_violin_plot(df_data, x_column="SkinThickness", title="SkinThickness Distribution (Violin Plot)")
        st.plotly_chart(plot_distribution('SkinThickness', 0))
        st.markdown('''We observe that            
-> "SkinThickness" values are concentrated around 0, there are outliers indicating unusual skin thickness measurements.                
-> Individuals with diabetes tend to have higher skin thickness values                   
-> Two distinct peaks in "SkinThickness" values, suggesting the presence of two subpopulations with different skin thickness characteristics
                    
                    ''')

    elif data_button == 'Insulin':
        histogram = create_histogram(df_data, x_column="Insulin", title="Insulin Distribution (Histogram)")
        boxplot = create_boxplot(df_data, x_column="Insulin", title="Insulin Distribution (Boxplot)")
        violin_plot = create_violin_plot(df_data, x_column="Insulin", title="Insulin Distribution (Violin Plot)")
        st.plotly_chart(plot_distribution('Insulin', 0))
        st.markdown('''We observe that             
-> A violin plot may show that insulin levels are bimodal, indicating that there are two distinct groups of individuals with either high or low insulin levels, with fewer individuals in the middle range.                       
-> The plots for Insulin are highly skewed.                
-> A histogram may show that the majority of individuals have insulin levels clustered around a specific range, with a long tail to the right. This could indicate that most individuals have normal insulin levels, but some have unusually high insulin levels.                              
-> Insulin's medians by the target are really different ! 102.5 for a healthy person and 169.5 for a diabetic person. It can suggest that the "Insulin" feature might be associated with diabetes.
                    
                    ''')
    elif data_button == 'BMI':
        histogram = create_histogram(df_data, x_column="BMI", title="BMI Distribution (Histogram)")
        boxplot = create_boxplot(df_data, x_column="BMI", title="BMI Distribution (Boxplot)")
        violin_plot = create_violin_plot(df_data, x_column="BMI", title="BMI Distribution (Violin Plot)")
        st.plotly_chart(plot_distribution('BMI', 0))
        st.markdown('''We observe that:           
-> Median (32.0) and Mean (31.9) of BMI are very close to each other. Thus, the distribution is more or less symmetric and uniform.        
-> People with diabetes tend to have more BMI.                  
-> The histogram shows a peak in the middle BMI range, it may suggest that most individuals in the dataset have a BMI in that range.        

                    ''')

    elif data_button == 'Age':
        histogram = create_histogram(df_data, x_column="Age", title="Age Distribution (Histogram)")
        boxplot = create_boxplot(df_data, x_column="Age", title="Age Distribution (Boxplot)")
        violin_plot = create_violin_plot(df_data, x_column="Age", title="Age Distribution (Violin Plot)")
        st.plotly_chart(plot_distribution('Age', 0))
        st.markdown('''We can observe that              
    -> the histogram shows a significant number of individuals in the dataset are between 40 and 60 years old, with a smaller number of individuals in other age groups.               
    -> The distribution of Age is skewed on the left side.              
    -> There are some outliers in the Box plot for Age.         
                   
                     ''')
    
    elif data_button == 'Outcome':
        fig, ax = plt.subplots(1, 2, figsize=(20, 7))
        sns.countplot(data=df_data, x="Outcome", ax=ax[0])
        df_data["Outcome"].value_counts().plot.pie(explode=[0.1, 0],autopct="%1.1f%%", labels=["No", "Yes"], shadow=True, ax=ax[1])
        st.pyplot(fig)
        st.markdown('''We observe from the above plot that:           
-> 65.1% patients in the dataset do NOT have diabetes.                   
-> 34.9% patients in the dataset has diabetes.
                    
                    ''')

    elif data_button == 'Diabetes Pedigree Function':
        histogram = create_histogram(df_data, x_column="DiabetesPedigreeFunction", title="Diabetes Pedigree Function Distribution (Histogram)")
        boxplot = create_boxplot(df_data, x_column="DiabetesPedigreeFunction", title="Diabetes Pedigree Function Distribution (Boxplot)")
        violin_plot = create_violin_plot(df_data, x_column="DiabetesPedigreeFunction", title="Diabetes Pedigree Function Distribution (Violin Plot)")
        st.plotly_chart(plot_distribution('DiabetesPedigreeFunction', 0))
        st.markdown('''We observe that:          
-> The histogram is higly skewed on the left side.        
-> There are many outliers in the Box plot.                 
-> Violin plot distribution is dense in the interval 0.0 - 1.0
                    
                    ''')

# Function for Multivariate analysis graphs with Oucome
def create_histplot(df, x_column, outcome_column = "Outcome", figsize=(20, 8), multiple="dodge", kde=True):
    fig, ax = plt.subplots(figsize=figsize)
    sns.histplot(data=df, x=x_column, hue=outcome_column, shrink=0.8, multiple=multiple, kde=kde, ax=ax)
    st.pyplot(fig)

# Multivariate Exploratory Data Analysis
def multi_eda():
    data_button = st.selectbox('Please choose preferred Multivariate visualization with Outcome', ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'Age', 'Diabetes Pedigree Function'])
    
    if data_button == 'Pregnancies':
        create_histplot(df_data, "Pregnancies")
        st.markdown('''''')

    elif data_button == 'Glucose':
        create_histplot(df_data, "Glucose")
        st.markdown('''From the above plot, we see a positive linear correlation.                      

-> As the value of Glucose increases, the count of patients having diabetes increases i.e. value of Outcome as 1, increases.           
-> Also, after the Glucose value of 125, there is a steady increase in the number of patients having Outcome of 1.               
-> There is a significant amount of positive linear correlation.
                    
                    ''')

    elif data_button == 'Blood Pressure':
        create_histplot(df_data, "BloodPressure")       
        st.markdown('''We observe that, Outcome and BloodPressure do NOT have a positive or negative linear correlation. The value of Outcome do not increase linearly as value of BloodPressure increases.However, for BloodPressure values greater than 82, count of patients with Outcome as 1, is more.
                    
                    ''')

    elif data_button == 'SkinThickness':
        create_histplot(df_data, "SkinThickness")

    elif data_button == 'Insulin':
        create_histplot(df_data, "Insulin")
        st.markdown('''A positive linear correlation is evident for Insuline''')
    elif data_button == 'BMI':
        create_histplot(df_data, "BMI")
        st.markdown('''A positive linear correlation is evident for BMI''')

    elif data_button == 'Age':
        create_histplot(df_data, "Age")
        st.markdown('''-> For Age greater than 35 years, the chances of patients having diabetes increases as evident from the plot i.e. The number of patients having diabetes is more than the number of people NOT having diabetes. But, it does not hold true for ages like 60+.                  
-> There is some positive linear correlation though.
                    
                    ''')

    elif data_button == 'Diabetes Pedigree Function':
        create_histplot(df_data, "DiabetesPedigreeFunction")
        st.markdown('There is some positive linear correlation of Pregnancies with Outcome.')

# Scatter plot between 2 columns
def plot_feat1_feat2(feat1, feat2) :  
    D = df_data[(df_data['Outcome'] != 0)]
    H = df_data[(df_data['Outcome'] == 0)]
    trace0 = go.Scatter(
        x = D[feat1],
        y = D[feat2],
        name = 'diabetic',
        mode = 'markers', 
        marker = dict(color = '#FFD700',
            line = dict(
                width = 1)))

    trace1 = go.Scatter(
        x = H[feat1],
        y = H[feat2],
        name = 'healthy',
        mode = 'markers',
        marker = dict(color = '#7EC0EE',
            line = dict(
                width = 1)))

    layout = dict(title = feat1 +" "+"vs"+" "+ feat2,
                  yaxis = dict(title = feat2,zeroline = False),
                  xaxis = dict(title = feat1, zeroline = False)
                 )
    plots = [trace0, trace1]
    fig = dict(data=plots, layout=layout)
    st.plotly_chart(fig)

# Hi Plot html saving for the dataset
def save_hiplot_to_html(exp):
    output_file = "hiplot_plot_1.html"
    exp.to_html(output_file)
    return output_file        

# Exploratory Data Analsis
def eda():
    st.sidebar.header('Exploratory Data Analysis')
    column_names = [col for col in df_data.columns]

    if st.checkbox('View Dataset'):
        st.caption("View the first 10 rows of the dataset")
        st.table(df_data.head(10))
        st.write('''The columns of this dataset are as follows:                  

-> Pregnancies — Number of times pregnant                    
-> GlucosePlasma — glucose concentration 2 hours in an oral glucose tolerance test            
-> Blood Pressure — Diastolic blood pressure (mm Hg)            
-> SkinThickness — Triceps skin-fold thickness (mm)           
-> Insulin — Two hours of serum insulin (mu U/ml)            
-> BMI — Body mass index (weight in kg/(height in m)²)             
-> Diabetes Pedigree Function — Diabetes pedigree function            
-> Age — Age in years                        
-> Outcome — Class variable (0-Non Diabetec or 1-Diabetec)''')
        
    if st.checkbox('Missing Values Handling'):
        missing_value_analysis()
    
    if st.checkbox('Univariate Analysis'):
        uni_eda()

    if st.checkbox('Multivariate Analysis - Relation Between "Outcome" and Other Variables'):
        multi_eda()

    if st.checkbox('Correlation'):
        st.write('Correlation matrix allows you to see which pairs have the highest correlation.')
        st.write('We select the columns with the highest and average correlation to perform subsequent multivariate analysis, aimed at feature extraction and dimensionality reduction.')
        sns.set_theme(style="white")
        st.caption("Correlation Matrix")
        corr = df_data.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        f, ax = plt.subplots(figsize=(20, 7))
        sns.heatmap(corr, mask=mask, square=True, annot=True, linewidths=.5, cbar_kws={"shrink": .6}, cmap="crest")
        st.pyplot(f) 

    if st.checkbox('Pairplot'):
        #fig = plt.figure(figsize=(20, 7))
        st.caption("Pairplot")
        st.pyplot(sns.pairplot(df_data, hue="Outcome"))
        
        st.markdown('')

    if st.checkbox('Multivariate Analysis'):
        st.subheader("Exploring the Relationship between column combinations with high correlation")
        plot_feat1_feat2("Insulin", "Glucose")
        st.markdown('Healthy people are oncentrated in between glucose levels 65 to 140 and insulene level 0 to 200')
        plot_feat1_feat2("Glucose", "Age")
        st.markdown('Healthy persons are concentrate with an age <= 30 and glucose <= 120')
        plot_feat1_feat2("Glucose", "BloodPressure")
        st.markdown('Healthy persons are concentrate with an blood pressure <= 80 and glucose <= 105. Diabetec people tend to have more Glucode levels.')
        plot_feat1_feat2("SkinThickness", "BMI")
        st.markdown('Healthy persons are concentrate with a BMI < 30 and skin thickness <= 20.')
        plot_feat1_feat2("Pregnancies", "Age")
        st.markdown('Healthy people below 30 have Pregnancies less than 6')

    if st.checkbox('Hi Plot'):
        st.write('This plot allows user to select required columns and visualize them using HiPlot. By systematically exploring the dataset, we can uncover relationships into how attributes may be correlated with the presence or absence of heart disease within specific age groups and clinical attribute ranges.')
        selected_columns = st.multiselect("Select columns to visualize", df_data.columns)
        selected_data = df_data[selected_columns]
        if not selected_data.empty:
            experiment = hip.Experiment.from_dataframe(selected_data)
            hiplot_html_file = save_hiplot_to_html(experiment)
            st.components.v1.html(open(hiplot_html_file, 'r').read(),width =900, height=600, scrolling=True)
        else:
            st.write("No data selected. Please choose at least one column to visualize.")
        
# Function for the Prediction of the reults using ML Models
def predict():
    st.sidebar.header('Diabetes Prediction')
    st.markdown('This trained dataset is originally from the Pima Indians Diabetes dataset. The objective is to predict based on diagnostic measurements whether a patient has diabetes.')
    
    name = st.text_input("Name:")
    pregnancy = st.number_input("No. of times pregnant:")
    st.markdown('Pregnancies: Number of times pregnant')

    glucose = st.number_input("Plasma Glucose Concentration :")
    st.markdown('Glucose: Plasma glucose concentration a 2 hours in an oral glucose tolerance test')

    bp =  st.number_input("Diastolic blood pressure (mm Hg):")
    st.markdown('BloodPressure: Diastolic blood pressure (mm Hg)')

    skin = st.number_input("Triceps skin fold thickness (mm):")
    st.markdown('SkinThickness: Triceps skin fold thickness (mm)')

    insulin = st.number_input("2-Hour serum insulin (mu U/ml):")
    st.markdown('Insulin: 2-Hour serum insulin (mu U/ml)')

    bmi = st.number_input("Body mass index (weight in kg/(height in m)^2):")
    st.markdown('BMI: Body mass index (weight in kg/(height in m)^2)')

    dpf = st.number_input("Diabetes Pedigree Function:")
    st.markdown('DiabetesPedigreeFunction: Diabetes pedigree function')

    age = st.number_input("Age:")
    st.markdown('Age: Age (years)')

    submit = st.button('Predict')
    st.markdown('Outcome: Class variable (0 or 1)')

    if submit:
        prediction = classifier.predict([[pregnancy, glucose, bp, skin, insulin, bmi, dpf, age]])
        if prediction == 0:
            st.write('Congratulation!',name,'You are not diabetic')
        else:
            st.write(name,", we are really sorry to say but it seems like you are Diabetic. But don't lose hope, we have suggestions for you:")
               
def main():
    new_title = '<p style="font-size: 42px;">Welcome to the Diabetes Prediction App!</p>'
    read_me_0 = st.markdown(new_title, unsafe_allow_html=True)
    read_me = st.markdown("""
    ## Diabetes Overview

Diabetes is a chronic condition characterized by elevated blood glucose levels. It occurs when the body either doesn't produce enough insulin or can't use it effectively. The three primary types are:

- **Type 1 Diabetes**: This type is commonly diagnosed in children and young adults, and those affected must take insulin regularly to survive. Here the immune system mistakenly attacks and destroys the insulin-producing cells in the pancreas. .

- **Type 2 Diabetes**: It is most prevalent in middle-aged and older individuals, but it can develop at any age and is the most common type of diabetes. Occurs when the body either doesn't produce sufficient insulin or cannot use it efficiently.

- **Gestational Diabetes**: Emerges during pregnancy and typically resolves after childbirth. However, women who experience gestational diabetes have an increased risk of developing type 2 diabetes later in life.

Building diabetes prediction applications are crucial for early detection, personalized care, and cost reduction. By analyzing diverse data sources, these models identify at-risk individuals, enabling timely interventions and tailored treatment plans. This proactive approach not only improves patient outcomes but also reduces healthcare costs, making it a valuable tool in addressing the global diabetes burden.
                        
The main aim is to make use of significant features, design a prediction algorithm using Machine learning and find the optimal classifier to give the closest result comparing to clinical outcomes. Doctors rely on common knowledge for treatment. When common knowledge is lacking, studies are summarized after some number of cases have been studied. But this process takes time, whereas if machine learning is used, the patterns can be identified earlier.Analyzing the details and understanding the patterns in the data can help in better decision-making resulting in a better quality of patient care. It can aid to understand the trends to The proposed method aims to focus on improvise the outcome of medical care, life expectancy, early detection, and identification of diabetis milletus disease at an initial stage and required treatment at an affordable cost.
    """)
    st.sidebar.title("Diabetes Predection")
    choice = st.sidebar.selectbox(
        "MODE", ("About", "Exploratory Data Analysis", "Predict Diabetes"))
    if choice == "Exploratory Data Analysis":
        read_me_0.empty()
        read_me.empty()
        eda()    
        st.sidebar.info("This App allows users to input their health information and receive an estimate of their risk for Diabetes. It could help them take necessary precautions and medication accordingly.")
        sidebar_placeholder = st.sidebar.empty()
    elif choice == "Predict Diabetes":
        read_me_0.empty()
        read_me.empty()
        predict()
        st.sidebar.info("This App allows users to input their health information and receive an estimate of their risk for Diabetes. It could help them take necessary precautions and medication accordingly.")
        sidebar_placeholder = st.sidebar.empty()
    elif choice == "About":
        print()
        st.sidebar.info("This App allows users to input their health information and receive an estimate of their risk for Diabetes. It could help them take necessary precautions and medication accordingly.")
        sidebar_placeholder = st.sidebar.empty()


if __name__ == '__main__':
    main()
