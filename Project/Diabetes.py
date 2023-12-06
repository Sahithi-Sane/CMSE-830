import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.offline as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.tools as tls
import plotly.figure_factory as ff
import squarify
import altair as alt
import plotly.express as px
import streamlit as st
import hiplot as hip
from plotly.subplots import make_subplots
py.init_notebook_mode(connected=True)
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
from yellowbrick.classifier import DiscriminationThreshold
import lightgbm as lgbm
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold
from scipy import interp
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, precision_recall_curve, auc, confusion_matrix, accuracy_score,make_scorer
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score




#------------------------------------------------------------------------------------------------------------------------------------------
st.title('Diabetis Dataset')
st.caption('Presented by Sahithi Sane')
st.divider()

# Load the dataset into the dataframe
df_data = pd.read_csv('https://raw.githubusercontent.com/Sahithi-Sane/CMSE-830/main/Project/diabetes.csv')
#csvFile = pd.read_csv('https://raw.githubusercontent.com/Sahithi-Sane/CMSE-830/main/Project/ensembling.csv')
df_temp = df_data

def load_weights_from_github(url):
    response = requests.get(url)
    if response.status_code == 200:
        return response.content
    else:
        st.error(f"Failed to fetch weights from {url}")
        return None

D = df_data[(df_data['Outcome'] != 0)]
H = df_data[(df_data['Outcome'] == 0)]

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

# Function for Multivariate analysis graphs with Outcome
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

    elif data_button == 'BloodPressure':
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

# Score table for LightGBM
def scores_table(model, subtitle):
    scores = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    res = []
    for sc in scores:
        scores = cross_val_score(model, X, y, cv = 5, scoring = sc)
        res.append(scores)
    df = pd.DataFrame(res).T
    df.loc['mean'] = df.mean()
    df.loc['std'] = df.std()
    df= df.rename(columns={0: 'accuracy', 1:'precision', 2:'recall',3:'f1',4:'roc_auc'})

    trace = go.Table(
        header=dict(values=['<b>Fold', '<b>Accuracy', '<b>Precision', '<b>Recall', '<b>F1 score', '<b>Roc auc'],
                    line = dict(color='#7D7F80'),
                    fill = dict(color='#a1c3d1'),
                    align = ['center'],
                    font = dict(size = 15)),
        cells=dict(values=[('1','2','3','4','5','mean', 'std'),
                           np.round(df['accuracy'],3),
                           np.round(df['precision'],3),
                           np.round(df['recall'],3),
                           np.round(df['f1'],3),
                           np.round(df['roc_auc'],3)],
                   line = dict(color='#7D7F80'),
                   fill = dict(color='#EDFAFF'),
                   align = ['center'], font = dict(size = 15)))

    layout = dict(width=800, height=400, title = '<b>Cross Validation - 5 folds</b><br>'+subtitle, font = dict(size = 15))
    fig = dict(data=[trace], layout=layout)

    py.iplot(fig, filename = 'styled_table')

# Function for data encoding
def preprocess_data(df):
    # Define target and feature columns
    target_col = ["Outcome"]
    cat_cols = df.nunique()[df.nunique() < 12].keys().tolist()
    cat_cols = [x for x in cat_cols]
    num_cols = [x for x in df.columns if x not in cat_cols + target_col]
    bin_cols = df.nunique()[df.nunique() == 2].keys().tolist()
    multi_cols = [i for i in cat_cols if i not in bin_cols]

    # Label encoding for binary columns
    le = LabelEncoder()
    for i in bin_cols:
        df[i] = le.fit_transform(df[i])

    # One-hot encoding for multi-value columns
    df = pd.get_dummies(data=df, columns=multi_cols)

    # Scaling numerical columns
    std = StandardScaler()
    scaled = std.fit_transform(df[num_cols])
    scaled = pd.DataFrame(scaled, columns=num_cols)

    # Drop original values and merge scaled values for numerical columns
    df_og = df.copy()
    df = df.drop(columns=num_cols, axis=1)
    df = df.merge(scaled, left_index=True, right_index=True, how="left")

    return df, df_og

# Encoding for the dataset
def encoding():
    st.markdown('''StandardScaler :
    - Standardize features by removing the mean and scaling to unit variance : Centering and scaling happen independently on each feature by computing the relevant statistics on the samples in the set. Mean and standard deviation are then stored to be used on later data using the transform method. Standardization of a dataset is a common requirement for many machine learning estimators: they might behave badly if the individual features do not more or less look like standard normally distributed data (e.g. Gaussian with 0 mean and unit variance)''')
    st.markdown('''
                   LabelEncoder : 
    - Encode labels with value between 0 and n_classes-1.
                ''')
    df_processed, df_og = preprocess_data(df_data)

    # Display processed data
    st.subheader("Processed Data")
    st.caption("View the first 10 rows of the dataset")
    st.table(df_processed.head(10))

# Plots for the models
def calculate_metrics_and_plots(model,train_X, train_y, test_X, test_y):
    # Train the classifier
    model.fit(train_X, train_y)

    # Predict on the test set
    y_pred_model = model.predict(test_X)

    # Calculate metrics
    ac = accuracy_score(test_y, y_pred_model)
    rc = roc_auc_score(test_y, y_pred_model)
    prec = precision_score(test_y, y_pred_model)
    rec = recall_score(test_y, y_pred_model)
    f1 = f1_score(test_y, y_pred_model)

    # Confusion Matrix
    cm = confusion_matrix(test_y, y_pred_model)

    # ROC Curve
    fpr, tpr, _ = roc_curve(test_y, model.predict_proba(test_X)[:, 1])
    roc_auc = auc(fpr, tpr)

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(test_y, model.predict_proba(test_X)[:, 1])
    pr_auc = auc(recall, precision)

    # Create Plots
    # Confusion Matrix Heatmap
    fig_cm = go.Figure()
    fig_cm.add_trace(go.Heatmap(z=cm[::-1], x=['Predicted 0', 'Predicted 1'], y=['Actual 1', 'Actual 0'],
                                colorscale='Viridis', showscale=False))
    fig_cm.update_layout(title='Confusion Matrix', xaxis=dict(title='Predicted Class'), yaxis=dict(title='Actual Class'))

    # ROC Curve
    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC curve (AUC={:.2f})'.format(roc_auc)))
    fig_roc.update_layout(title='Receiver Operating Characteristic (ROC) Curve',
                          xaxis=dict(title='False Positive Rate'),
                          yaxis=dict(title='True Positive Rate'),
                          showlegend=True)

    # Precision-Recall Curve
    fig_pr = go.Figure()
    fig_pr.add_trace(go.Scatter(x=recall, y=precision, mode='lines', name='Precision-Recall curve (AUC={:.2f})'.format(pr_auc)))
    fig_pr.update_layout(title='Precision-Recall Curve',
                         xaxis=dict(title='Recall'),
                         yaxis=dict(title='Precision'),
                         showlegend=True)

    # Metrics Bar Graph
    metrics_labels = ['Accuracy', 'ROC AUC', 'Precision', 'Recall', 'F1-Score']
    metrics_values = [ac, rc, prec, rec, f1]

    fig_metrics = go.Figure()
    fig_metrics.add_trace(go.Bar(x=metrics_labels, y=metrics_values, name='Metrics'))
    fig_metrics.update_layout(barmode='group', xaxis=dict(title='Metrics'), yaxis=dict(title='Value'))

    return fig_cm, fig_roc, fig_pr, fig_metrics

# Model training, metrics
def model_analysis():
    X = df_data.drop('Outcome', axis=1)
    y = df_data['Outcome']
    train_X,test_X,train_y,test_y=train_test_split(X,y,test_size=0.2)
    st.markdown("There are 9 different models and their performance metrics to compare and see which is best for the prediction.")

    # Load or generate your data here
    # For demonstration purposes, let's create some random data
    np.random.seed(42)
    train_X = np.random.rand(100, 5)
    train_y = np.random.randint(0, 2, 100)
    test_X = np.random.rand(50, 5)
    test_y = np.random.randint(0, 2, 50)

    # Model selection
    data_button = st.selectbox('Please choose preferred Model to get the anaysis', ["Logistic Regression", "Random Forest","Support Vector Machine","K Nearest Neighbour","Naive Bayes","Gradient Boosting Classifier","Decision Tree","XG Boost","LightGBM"])
    if  data_button =="Logistic Regression":
        model = LogisticRegression()
    elif  data_button == "Random Forest":
        model = RandomForestClassifier()
    elif  data_button == "Support Vector Machine":
        model = SVC(kernel='linear')
    elif  data_button == "K Nearest Neighbour":
        model = KNeighborsClassifier(n_neighbors=3)
    elif  data_button == "Naive Bayes":
        model = GaussianNB()
    elif  data_button == "Gradient Boosting Classifier":
        model = GradientBoostingClassifier(n_estimators=50,learning_rate=0.2)
    elif  data_button == "Decision Tree":
        model = RandomForestClassifier(n_estimators = 11, criterion = 'entropy', random_state = 42)
    elif  data_button == "XG Boost":
        model = XGBClassifier()
    elif  data_button == "LightGBM":
        model = XGBClassifier()    
    else:
        st.error("Invalid model selection.")

    # Calculate metrics and create plots
    fig_cm, fig_roc, fig_pr, fig_metrics = calculate_metrics_and_plots(model, train_X, train_y, test_X, test_y)

    # Display Plots
    st.subheader("Confusion Matrix")
    st.plotly_chart(fig_cm)

    st.subheader("ROC Curve")
    st.plotly_chart(fig_roc)

    st.subheader("Precision-Recall Curve")
    st.plotly_chart(fig_pr)

    st.subheader("Metrics Bar Graph")
    st.plotly_chart(fig_metrics)


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
    
    if st.checkbox('StandardScaler and LabelEncoder'):
        encoding()

    if st.checkbox('Models'):
        model_analysis()
    
    if st.checkbox('Ensembling'):
        st.write('Heatmap and other metrix for Ensembled model with Maximum Voting of 9 Models.')
        
        

def bio():
    fgd

# Function for the Prediction of the reults using ML Models
def predict():
    st.sidebar.header('Diabetes Prediction')
    st.markdown('This trained dataset is originally from the Pima Indians Diabetes dataset. The objective is to predict based on diagnostic measurements whether a patient has diabetes.')
    
    name = st.text_input("Name:")

    pregnancy = st.number_input("No. of times pregnant:")
    st.markdown('Pregnancies: Number of times pregnant')

    glucose = st.number_input("Plasma Glucose Level :")
    st.markdown('Glucose: Plasma glucose concentration a 2 hours in an oral glucose tolerance test')

    bp =  st.number_input("Blood pressure (mm Hg):")
    st.markdown('BloodPressure: Diastolic blood pressure (mm Hg)')

    skin = st.number_input("Triceps skin fold thickness (mm):")
    st.markdown('SkinThickness: Triceps skin fold thickness (mm)')

    insulin = st.number_input("Serum insulin (mu U/ml):")
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
        X = df_data.drop('Outcome', axis=1)
        y = df_data['Outcome']
        train_X,test_X,train_y,test_y=train_test_split(X,y,test_size=0.2)
        model = RandomForestClassifier()
        model.fit(train_X,train_y)
        prediction = model.predict([[pregnancy, glucose, bp, skin, insulin, bmi, dpf, age]])
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
        "MODE", ("About", "Exploratory Data Analysis", "Predict Diabetes", "Bio"))
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
    elif choice == "Bio":
        read_me_0.empty()
        read_me.empty()
        bio()    
        st.sidebar.info("This App allows users to input their health information and receive an estimate of their risk for Diabetes. It could help them take necessary precautions and medication accordingly.")
        sidebar_placeholder = st.sidebar.empty()
if __name__ == '__main__':
    main()
