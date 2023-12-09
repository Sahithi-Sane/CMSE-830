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
from PIL import Image
import pickle as pkl
import altair as alt
import plotly.express as px
import streamlit as st
import hiplot as hip
from plotly.subplots import make_subplots
py.init_notebook_mode(connected=True)
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
import lightgbm as lgbm
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold
from scipy import interp
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, precision_recall_curve, auc, accuracy_score,make_scorer
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
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from yellowbrick.classifier import DiscriminationThreshold



#------------------------------------------------------------------------------------------------------------------------------------------
st.title('DiaPred - Diabetes Prediction Application')
st.caption('Presented by Sahithi Sane')
st.divider()

# Load the dataset into the dataframe
df_data = pd.read_csv('https://raw.githubusercontent.com/Sahithi-Sane/CMSE-830/main/Project/diabetes.csv')
DATA_URL = ('https://raw.githubusercontent.com/Sahithi-Sane/CMSE-830/main/Project/ensembling.csv')
@st.cache
def load_data():
    data = pd.read_csv(DATA_URL, encoding='latin1')
    return data
csvFile = load_data()
df_temp = df_data

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
def calculate_metrics_and_plots_interactive(model, test_X, test_y, train_X,train_y):
    # Predict on the test set
    model.fit(train_X,train_y)
    y_pred = model.predict(test_X)
    y_pred_proba = model.predict_proba(test_X)[:, 1]

    # Calculate confusion matrix
    cm = confusion_matrix(test_y, y_pred)

    # Calculate ROC curve and AUC
    fpr, tpr, _ = roc_curve(test_y, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    # Calculate precision-recall curve and AUC
    precision, recall, _ = precision_recall_curve(test_y, y_pred_proba)
    pr_auc = auc(recall, precision)

    # Calculate additional metrics
    accuracy = accuracy_score(test_y, y_pred)
    precision_score_val = precision_score(test_y, y_pred)
    recall_score_val = recall_score(test_y, y_pred)
    f1_score_val = f1_score(test_y, y_pred)

    # Plot interactive confusion matrix
    fig, ax = plt.subplots()
    cax = ax.matshow(cm, cmap='Blues')
    plt.title('Confusion Matrix')
    plt.colorbar(cax)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), va='center', ha='center', fontsize=12)

    # Plot interactive ROC curve
    fig_roc, ax_roc = plt.subplots()
    ax_roc.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
    ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax_roc.set_title('Receiver Operating Characteristic')
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.legend(loc='lower right')

    # Plot interactive precision-recall curve
    fig_pr, ax_pr = plt.subplots()
    ax_pr.plot(recall, precision, color='blue', lw=2, label='Precision-Recall curve (area = {:.2f})'.format(pr_auc))
    ax_pr.set_title('Precision-Recall Curve')
    ax_pr.set_xlabel('Recall')
    ax_pr.set_ylabel('Precision')
    ax_pr.legend(loc='upper right')

    # Plot bar graph for metrics
    metrics_labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    metrics_values = [accuracy, precision_score_val, recall_score_val, f1_score_val]

    fig_metrics, ax_metrics = plt.subplots()
    ax_metrics.bar(metrics_labels, metrics_values, color=['lightblue', 'lightgreen', 'lightcoral', 'lightsalmon'])
    ax_metrics.set_title('Metrics')
    ax_metrics.set_ylabel('Score')

    return fig, fig_roc, fig_pr, fig_metrics

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
        with st.expander("Understand what Logistic Regression is and how it works"):
            st.write("Logistic Regression is a statistical technique used for binary classification. It estimates the probability of an observation belonging to one of two classes. It uses a sigmoid function to transform input features into probabilities and draws a decision boundary to separate classes. The model learns from labeled data, assigns importance to features, and is evaluated based on its predictive performance using metrics like accuracy, precision, recall, and F1-score.")                                                    
        model = LogisticRegression()
    elif  data_button == "Random Forest":
        st.write("Random Forest is a powerful machine learning technique that builds numerous decision trees using random subsets of data and features. These individual trees work together by voting (for classification problems) or averaging (for regression tasks) to produce predictions. It's highly effective, particularly with large datasets, as it mitigates overfitting issues commonly found in single decision trees. Due to its ability to create diverse models and combine their outputs, Random Forest is known for its accuracy and robustness across different applications in machine learning")
        model = RandomForestClassifier()
    elif  data_button == "Support Vector Machine":
        st.markdown("Support Vector Machine (SVM) is a supervised machine learning algorithm used for classification and regression tasks. It works by finding the hyperplane that best separates different classes in the feature space, maximizing the margin between them. SVM is effective in high-dimensional spaces and can handle both linear and non-linear relationships through the use of kernel functions.")
        model = SVC(kernel='linear')
    elif  data_button == "K Nearest Neighbour":
        with st.expander("Understand what K-Nearest Neighbors is and how it works"):
            st.write("K-Nearest Neighbors (KNN) is a machine learning algorithm used for classification and regression. It predicts the class or value of a new data point by considering the majority (for classification) or averaging (for regression) the 'K' nearest data points in the training set based on a chosen distance measure, typically Euclidean distance. Its simplicity makes it easy to understand, but it can be computationally intensive for large datasets during the prediction phase. The choice of 'K' influences the model's performance.")
        model = KNeighborsClassifier(n_neighbors=3)
    elif  data_button == "Naive Bayes":
        with st.expander("Understand what Naive Bayes is and how it works"):
            st.write("Naive Bayes is a classification algorithm based on Bayes' theorem. It assumes features are normally distributed and independent. It calculates the probability that a data point belongs to a particular class using Gaussian (normal) distributions for numeric features. This method is commonly used in text classification, medical diagnosis, and similar tasks where feature independence and Gaussian distribution hold. Despite its simplicity, it's often effective and computationally efficient, especially with smaller datasets.")       
        model = GaussianNB()
    elif  data_button == "Gradient Boosting Classifier":
        with st.expander("Gradient Boosting Classifier is an ensemble machine learning algorithm that builds a strong predictive model by combining multiple weak learners, typically decision trees, sequentially. It works by fitting each tree to the residuals (errors) of the preceding one, adjusting the model iteratively to minimize the overall prediction errors. This iterative process strengthens the model's ability to capture complex relationships in the data, resulting in a powerful and accurate classifier."):
            st.write("")
        model = GradientBoostingClassifier(n_estimators=50,learning_rate=0.2)
    elif  data_button == "Decision Tree":
        with st.expander("Understand what Naive Bayes is and how it works"):
            st.write("Decision tree modeling is a machine learning technique that creates a tree-like structure to make decisions based on input data. It selects features to split the data into subsets, aiming to make the subsets as homogeneous as possible regarding the target variable. This process continues recursively until a stopping criterion is met. When new data is given, the model traverses the tree to predict the outcome based on the input features. Advantages include interpretability and the ability to capture non-linear relationships.")
            st.write(" ")
            st.write("However, decision trees can also suffer from certain limitations like overfitting (creating overly complex trees that perform well on training data but poorly on unseen data), instability with small variations in data, and sometimes not achieving the highest predictive accuracy compared to other algorithms.")                                    
        model = RandomForestClassifier(n_estimators = 11, criterion = 'entropy', random_state = 42)
    elif  data_button == "XG Boost":
        with st.expander("Understand what XG Boost is and how it works"):
            st.write("XGBoost (Extreme Gradient Boosting) is a powerful machine learning algorithm that belongs to the ensemble learning family. It builds a series of decision trees and combines their predictions to improve accuracy and reduce overfitting. XGBoost employs a gradient boosting framework, optimizing the model by minimizing the residuals of the previous trees, resulting in a highly efficient and effective algorithm for both classification and regression tasks.")
        model = xgb = XGBClassifier()
    elif  data_button == "LightGBM":
        with st.expander("Understand what LightGBM is and how it works"):
            st.write("LightGBM is a gradient boosting framework that efficiently trains decision tree ensembles. It employs a histogram-based approach for binning continuous features, reducing memory usage and speeding up training. LightGBM uses a leaf-wise tree growth strategy, optimizing for computational efficiency and scalability, making it particularly well-suited for large datasets.")
        opt_parameters =  grid_search.best_params_
        model = lgbm.LGBMClassifier(**opt_parameters) 
    else:
        st.error("Invalid model selection.")

    # Calculate metrics and create plots
    fig_cm, fig_roc, fig_pr, fig_metrics = calculate_metrics_and_plots_interactive(model, test_X, test_y, train_X,train_y)
    # Display figures
    st.subheader('Confusion Matrix')
    st.pyplot(fig_cm)

    st.subheader('ROC Curve')
    st.pyplot(fig_roc)

    st.subheader('Precision-Recall Curve')
    st.pyplot(fig_pr)

    st.subheader('Metrics Bar Graph')
    st.pyplot(fig_metrics)
    

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
        fig = plt.figure()
        cm = confusion_matrix(csvFile['Actual'], csvFile['Ensembling'])

        # Display the confusion matrix heatmap using Seaborn
        sns.heatmap(pd.DataFrame(cm), annot=True, fmt='g', cmap='Blues', cbar=False)
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')

        # Display the confusion matrix heatmap using Streamlit
        st.pyplot(fig)
        # Display accuracy
        ac = accuracy_score(csvFile['Actual'], csvFile['Ensembling'])
        st.subheader(f"Accuracy: {ac:.2f}")

def bio():
    st.write("Hi there! I am Sahithi Sane, currently pursuing Master's in Data Science at Michigan State University. I'm am Pythonista, Data Science and Artificial intelligence enthusiast, passionate about extracting insights from data using various analytical tools and techniques. ")
    st.write(" ")
    st.write("I have a strong academic background in data science, and mathematics. Have industrial experience in DataOps, MLOps, extensive dataset analysis, and predictive modeling. Adept at utilizing open-source technology, strong interpersonal skills, analytical skills, and a collaborative problem solver.")
    st.write(" ")
    st.write("Thriving on challenges, I engage in impactful endeavors that matter. When I'm not diving into data, I love spending time in nature, capturing moments through photography, and honing my culinary skills by experimenting with different cuisines.")
    st.write(" ")
    st.write("I'm excited to be a part of this project because it aligns perfectly with my passion for leveraging data to create meaningful solutions. I believe that by applying data science principles, we can solve real-world problems and make a positive difference.")
    st.write(" ")
    st.write("Feel free to reach out if you have any questions or just want to discuss data science, philosophy, or anything else that piques your curiosity!")
    st.write(" ")
    '''
    file = 'https://github.com/Sahithi-Sane/CMSE-830/blob/main/Project/me.jpeg'
    if file is not None:
        image = Image.open(file)
        st.image(image,
                 caption=f"You amazing image has shape",
                use_column_width=True,)
   # st.image('me.jpeg', width=300)
   '''
# Function for the Prediction of the reults using ML Models
def predict():
    st.sidebar.header('Diabetes Prediction')
    st.markdown('This trained dataset is originally from the Pima Indians Diabetes dataset. The objective is to predict based on diagnostic measurements whether a patient has diabetes.')
    
    name = st.text_input("Name:")

    st.markdown('Pregnancies: Number of times pregnant')
    pregnancy = st.number_input("No. of times pregnant:")
    
    st.markdown('Glucose: Plasma glucose concentration a 2 hours in an oral glucose tolerance test(Should be aroung 150 generally)')
    glucose = st.number_input("Plasma Glucose Level :")

    st.markdown('BloodPressure: Diastolic blood pressure (mm Hg)')
    bp =  st.number_input("Blood pressure (mm Hg):")
    
    st.markdown('SkinThickness: Triceps skin fold thickness (mm) - Women :23.6 ± 7.5 mm and Men :14.3 ± 6.8 mm')
    skin = st.number_input("Triceps skin fold thickness (mm):")
    
    st.markdown('Insulin: 2-Hour serum insulin (mu U/ml) - lower than 140 mg/dL')
    insulin = st.number_input("Serum insulin (mu U/ml):")
    

    bmi = st.number_input("Body mass index (weight in kg/(height in m)^2):")
    st.markdown('BMI: Body mass index (weight in kg/(height in m)^2)')

    dpf = st.number_input("Diabetes Pedigree Function - Diabetes Pedigree Function is a positively skewed variable with no zero values")
    st.markdown('DiabetesPedigreeFunction: Diabetes pedigree function')

    age = st.number_input("Age:")
    st.markdown('Age: Age (years)')

    submit = st.button('Predict')
    
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
        st.sidebar.write(
            "Welcome to a journey of self-discovery through your heart's story! 🌟\n\n"
            "Have you ever wondered what your heart health says about you?"
            )
        sidebar_placeholder = st.sidebar.empty()
    elif choice == "Predict Diabetes":
        read_me_0.empty()
        read_me.empty()
        predict()
        st.sidebar.write(
            "Welcome to a journey of self-discovery through your heart's story! 🌟\n\n"
            "Have you ever wondered what your heart health says about you?"
            )
        st.sidebar.info("This App allows users to input their health information and receive an estimate of their risk for Diabetes. It could help them take necessary precautions and medication accordingly.")
        sidebar_placeholder = st.sidebar.empty()
    elif choice == "About":
        print()
        st.sidebar.write(
            "Welcome to a journey of self-discovery through your heart's story! 🌟\n\n"
            "Have you ever wondered what your heart health says about you?"
            )
        st.sidebar.info("This App allows users to input their health information and receive an estimate of their risk for Diabetes. It could help them take necessary precautions and medication accordingly.")
        sidebar_placeholder = st.sidebar.empty()
    elif choice == "Bio":
        read_me_0.empty()
        read_me.empty()
        bio()    
        st.sidebar.write(
            "Welcome to a journey of self-discovery through your heart's story! 🌟\n\n"
            "Have you ever wondered what your heart health says about you?"
            )
        st.sidebar.info("This App allows users to input their health information and receive an estimate of their risk for Diabetes. It could help them take necessary precautions and medication accordingly.")
        sidebar_placeholder = st.sidebar.empty()
if __name__ == '__main__':
    main()
