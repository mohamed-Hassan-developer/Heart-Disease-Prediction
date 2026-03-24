
import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV, cross_val_predict ,cross_validate
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,MinMaxScaler,FunctionTransformer
from sklearn.impute import SimpleImputer

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier



html_title = """<h1 style="color:white;text-align:center;"> <span style="color:red">Heart Disease </span> Risk Exploratory Data Analysis & Prediction</h1>"""
st.markdown(html_title, unsafe_allow_html=True)
    # Set Title 
st.set_page_config(layout='wide', page_title= 'Heart Disease Risk EDA & HD Prediction',page_icon='💔')

page = st.sidebar.radio('Page', ['Home','Dash Board', 'Statistics', 'Dynamic Reports','Heart Disease Prediction'])

#@st.cache_data
#def load_data():
df = pd.read_csv('cleaned_df.csv', index_col= 0)
ML_df = pd.read_csv('cleaned_ML_df.csv', index_col= 0)
importance_df=pd.read_csv('importance.csv',index_col=0)
 #return df

#try:
#    df=load_data()

if page == 'Home':



    # Insert Image
    col1, col2, col3 = st.columns([1,2,1])

    
    col2.image(width=1000,image='https://www.riversidehealthcare.org/sites/default/files/healthcurrents/GettyImages-1344030014.jpg')


    st.header('Dataset Overview')
    col1, col2 = st.columns([2,1])

    col1.dataframe(df,height=1100)

        # Create table of column descriptions
    data = {
        "Column Name": [
            "Gender", "Age", "Age_Segment", "Blood Pressure",
            "Blood_Pressure_Ranges", "High Blood Pressure", "Stress Level",
            "Cholesterol Level", "Low HDL Cholesterol", "High LDL Cholesterol","Exercise Habits","Smoking","Diabetes","Sugar Consumption",
            "Fasting Blood Sugar", "BMI", "BMI categories", "Alcohol Consumption",
            "Sleep Hours", "Sleep_Type", "Triglyceride Level", "trigly_level",
            "CRP Level", "CRP_Group", "Homocysteine Level",
            "Homocysteine_Category", "Family Heart Disease",
            "Heart Disease Status", "Heart Disease Binary"
        ],
        "Description": [
            "Biological sex of the individual",
            "Age in years",
            "Categorized age segment",
            "Systolic blood pressure (mm/Hg)",
            "Blood pressure classification",
            "Whether the person has high BP",
            "Self-reported stress level",
            "Total cholesterol (mg/dL)",
            "Indicates low HDL cholesterol",
            "Indicates High HDL cholesterol",
            "Exercise habits (Low, Medium, High)",
            "Smoker or not (Yes or No)",
            "Diabetes or not (Yes or No)",
            "Daily sugar intake level",
            "Fasting blood glucose level",
            "Body Mass Index",
            "BMI classification",
            "Alcohol consumption level",
            "Average daily sleep duration",
            "Sleep quality category",
            "Triglyceride level (mg/dL)",
            "Triglyceride range classification",
            "C-Reactive Protein level",
            "CRP classification",
            "Homocysteine level",
            "Homocysteine category",
            "Family history of heart disease",
            "Heart disease type/status",
            "Binary heart disease indicator"
        ]
    }

    desc_df = pd.DataFrame(data)

    # Display table
    col2.subheader("📝 Column Descriptions")
    col2.table(desc_df)

elif page == 'Statistics':

    #col1, col2 = st.columns([1,1])

    #with col1:
    cat_col = [
        'Heart Disease Status','Gender','Age_Segment','Blood_Pressure_Ranges',
        'Stress Level','Low HDL Cholesterol','High LDL Cholesterol',
        'Exercise Habits','Smoking','Diabetes','Sugar Consumption','BMI categories',
        'Alcohol Consumption','Sleep_Type','trigly_level','CRP_Group',
        'Homocysteine_Category','Family Heart Disease'
    ]

    st.title("Categorical Analysis for Heart Disease")

    for col in cat_col:
        result = df[df['Heart Disease Status'] == 'Yes'].groupby(col)['Per'].sum().round(2).reset_index().sort_values('Per', ascending=False)
        
        with st.expander(f"📊 {col}"):
            col1, col2 = st.columns([1,1])

            #st.dataframe(result)
            col1.dataframe(result)
            Chart_Type=col1.radio('Chart Type :',options=['Histrogram','Pie'],key=col)
            if Chart_Type=='Histrogram':

                col2.plotly_chart(px.histogram(result, x=col, y='Per',text_auto= True, title=f"{col}"))
            else :
                col2.plotly_chart(px.pie(result, names=col, values='Per', title=f"{col}"))


elif page == 'Dash Board':

    cat_col = [
        'Heart Disease Status','Gender','Age_Segment','Blood_Pressure_Ranges',
        'Stress Level','Low HDL Cholesterol','High LDL Cholesterol',
        'Exercise Habits','Smoking','Diabetes','Sugar Consumption','BMI categories',
        'Alcohol Consumption','Sleep_Type','trigly_level','CRP_Group',
        'Homocysteine_Category','Family Heart Disease'
    ]

    st.title("Categorical Analysis for Heart Disease")

    for col in cat_col:
        result = df[df['Heart Disease Status'] == 'Yes'].groupby(col)['Per'].sum().round(2).reset_index().sort_values('Per', ascending=False)
        
        col1, col2 = st.columns([1,1])
        
        col2.dataframe(result)
        Chart_Type=col2.radio('Chart Type :',options=['Histrogram','Pie'],key=col)

        if Chart_Type=='Histrogram':
            col1.plotly_chart(px.histogram(result, x=col, y='Per',text_auto= True, title=f"{col}"))
        else :
            col1.plotly_chart(px.pie(result, names=col, values='Per', title=f"{col}"))

        
        co1,col2,col3 =st.columns([1,2,1])

        col2.write(f"### 💗 ▂▃▅▇ {col.upper()} ANALYSIS ▇▅▃▂ 💗")
        st.write("-----")


elif page =='Dynamic Reports' :

    col1, col2 = st.columns([3,1])

    All_gender =  ['Choose'] + df.Gender.unique().tolist() 
    Gender = st.sidebar.selectbox('Gender', All_gender)

    age_group=st.sidebar.selectbox('Age Group', ['All Age Groups','Teenager', 'Young Adult','Adult', 'Middle-Aged','Senior'])

    Boold_pressure_ranges =  ['All Ranges'] + ['Normal  [80-120]','Elevated  [120-129]','Stage 1 Hypertension  [130-139]','Stage 2 Hypertension [140 & above]'] 
    
    Boold_pressure_ranges = st.sidebar.selectbox('Blood Pressure Ranges', Boold_pressure_ranges)

    Alcohol_Consumption =  ['All'] +  ['Most likly Never','Low','Medium','High']
    
    Alcohol_Consumption = st.sidebar.selectbox('Alcohol Consumption', Alcohol_Consumption)

    BMI_categories =  ['All Categories'] + ['Underweight','Normal weight','Overweight','Obesity'] 
    
    BMI_categories = st.sidebar.selectbox('BMI categories', BMI_categories)

    CRP_Group =  ['All Groups'] + ['Normal/Low', 'Marked Elevation','Moderate Elevation']
    
    CRP_Group = st.sidebar.selectbox('CRP Group', CRP_Group)    
    
    Homocysteine_Category =  ['All Categories'] + ['Marked Elevation', 'Moderate Elevation','Severe Elevation'] 
    
    Homocysteine_Category = st.sidebar.selectbox('Homocysteine categories', Homocysteine_Category)

    Stress=st.sidebar.radio('Stess Level', options=['All'] + ['Low','Medium','High'],horizontal=True )

    Exercise_Habits=st.sidebar.radio('Exercise Habits', options=['All'] +  ['Low','Medium','High'],horizontal=True )

    Smoker=st.sidebar.radio('Smoker',options=['All','No','Yes'],horizontal=True)

    LDL_Cholesterol=st.sidebar.radio('Cholesterol',options=['All','No','Yes'],horizontal=True)

    Diabetes=st.sidebar.radio('Diabetes',options=['All','No','Yes'],horizontal=True)

    Family_Heart_Disease=st.sidebar.radio('Family Heart History',options=['All','No','Yes'],horizontal=True)

    Sleep=st.sidebar.radio('Sleep Type',options=['Any','Normal', 'Light', 'Deep'],horizontal=True)

    trigly_level = st.sidebar.radio('Triglyceride Level', ['All'] + ['Normal','High','Borderline'],horizontal=True)




    dF_select=col2.multiselect('Data Frame',df.drop(columns=['Per','Heart Disease Status','High Blood Pressure'],axis=1).columns,max_selections=26,default=df.drop(columns=['Per','Heart Disease Status','High Blood Pressure'],axis=1).columns)

    if dF_select != '':

        df_filtered_col = ['Heart Disease Status'] + dF_select

        df_filtered=df[df['Heart Disease Status']=='Yes'].groupby(df_filtered_col)['Per'].sum()
        df_filtered=df_filtered.reset_index().sort_values(by='Per',ascending=False)

    
    if Gender != 'Choose' and 'Gender' in df_filtered_col:

        df_filtered = df_filtered[df_filtered['Gender'] == Gender]

    if Smoker != 'All' and 'Smoking' in df_filtered_col:

        df_filtered = df_filtered[df_filtered['Smoking'] == Smoker]

    if age_group != 'All Age Groups' and 'Age_Segment' in df_filtered_col:

        df_filtered = df_filtered[df_filtered['Age_Segment'] == age_group]

    if Boold_pressure_ranges != 'All Ranges' and 'Blood_Pressure_Ranges' in df_filtered_col:

        df_filtered = df_filtered[df_filtered['Blood_Pressure_Ranges'] == Boold_pressure_ranges]


    if Alcohol_Consumption != 'All' and 'Alcohol Consumption' in df_filtered_col:

        df_filtered = df_filtered[df_filtered['Alcohol Consumption'] == Alcohol_Consumption]       


    if BMI_categories != 'All Categories' and 'BMI categories' in df_filtered_col:

        df_filtered = df_filtered[df_filtered['BMI categories'] == BMI_categories]  

    if CRP_Group != 'All Groups' and 'CRP_Group' in df_filtered_col:

        df_filtered = df_filtered[df_filtered['CRP_Group'] == CRP_Group]  

    if Homocysteine_Category != 'All Categories' and 'Homocysteine_Category' in df_filtered_col:

        df_filtered = df_filtered[df_filtered['Homocysteine_Category'] == Homocysteine_Category]  

    if Stress != 'All' and 'Stress Level' in df_filtered_col:

        df_filtered = df_filtered[df_filtered['Stress Level'] == Stress]  
 
    if Exercise_Habits != 'All' and 'Exercise Habits' in df_filtered_col:

        df_filtered = df_filtered[df_filtered['Exercise Habits'] == Exercise_Habits]     

    if Diabetes != 'All' and 'Diabetes' in df_filtered_col:

        df_filtered = df_filtered[df_filtered['Diabetes'] == Diabetes]  

    if Sleep != 'Any' and 'Sleep_Type' in df_filtered_col:

        df_filtered = df_filtered[df_filtered['Sleep_Type'] == Sleep]  

    if trigly_level != 'All' and 'trigly_level' in df_filtered_col:

        df_filtered = df_filtered[df_filtered['trigly_level'] == trigly_level] 


    if Family_Heart_Disease != 'All' and 'Family Heart Disease' in df_filtered_col:

        df_filtered = df_filtered[df_filtered['Family Heart Disease'] == Family_Heart_Disease] 

    if LDL_Cholesterol != 'All' and 'High LDL Cholesterol' in df_filtered_col:

        df_filtered = df_filtered[df_filtered['High LDL Cholesterol'] == LDL_Cholesterol] 


    if dF_select != '':
        col1.dataframe(df_filtered)
        st.plotly_chart(px.histogram(df_filtered, x=dF_select, y='Per',barmode='overlay',height=600,text_auto= True))
            
    else :
        col1.dataframe(df)

elif page =='Heart Disease Prediction' :

    colA ,colB = st.columns([3,1])

    col1, col2, col3 ,col4 = colA.columns([1,1,1,1])

    col111,col112 =colA.columns([6,1])

    col5, col6, col7 ,col8 = colA.columns([1,1,1,1])

    col113,col114 =colA.columns([6,1])

    col9, col10 ,col11,col12 = colA.columns([1,1,1,1])

    col115,col116 =colA.columns([6,1])

    col13, col14 ,col15 ,col16 = colA.columns([1,1,1,1])

    col117,col118 =colA.columns([6,1])

    col17, col18, col19 ,col20 = colA.columns([1,1,1,1])

    col119,col120 =colA.columns([6,1])

    Age_pre = col1.slider('Age',step=1)

    Gender_per = col2.radio('Gender', df.Gender.unique(),horizontal=True)

    Smoker_pre = col3.radio('Smoker',options=['No','Yes'],horizontal=True)

    Family_Heart_Disease_pre =col4.radio('Family Heart History',options=['No','Yes'],horizontal=True)

    col111.write("-----")


    Diabetes_pre =col5.radio('Diabetes',options=['No','Yes'],horizontal=True)

    Sugar_Consumption_pre = col6.radio('Sugar Consumption', options=['Low','Medium','High'],horizontal=True )

    Fasting_Blood_Sugar_pre = col7.number_input('Fasting Blood Sugar',min_value=40)

    BMI_pre = col8.number_input('BMI',min_value=15.000,max_value=35.000)

    col113.write("-----")

    Stress_pre =col9.radio('Stess Level', options=['Low','Medium','High'],horizontal=True )

    High_Blood_Pressure_pre =col10.radio('High Blood Pressure',options=['No','Yes'],horizontal=True)

    Blood_Pressure_pre = col11.number_input('Blood Pressure' )

    Sleep_Hours_pre = col12.number_input('Sleep Hours',min_value=1.000,max_value=23.000)



    col115.write("-----")


    Low_HDL_Cholesterol_pre = col13.radio('Low HDL Cholesterol',options=['No','Yes'],horizontal=True)
    
    High_LDL_Cholesterol_pre = col14.radio('High LDL Cholesterol',options=['No','Yes'],horizontal=True)

    Cholesterol_Level_pre = col15.number_input('Cholesterol Level',min_value=100.000)

    Alcohol_Consumption_pre = col16.selectbox('Alcohol Consumption', ['Low','Medium','High'])

    col117.write("-----")


    Exercise_Habits_pre =col17.radio('Exercise Habits', ['Low','Medium','High'],horizontal=True )

    Triglyceride_Level_pre = col18.number_input('Triglyceride Level',min_value=150.000,max_value=500.000)

    CRP_Level_pre = col19.number_input('CRP Level',min_value=0.1)

    Homocysteine_Level_pre = col20.number_input('Homocysteine Level',min_value=5.000)

    col119.write("-----")

    


    # Create Predicted row
    input_columns = ML_df.columns.drop('Heart Disease Status')
    predicted_data = pd.DataFrame(columns= input_columns, data= [[Age_pre,Gender_per, Blood_Pressure_pre, Cholesterol_Level_pre, Exercise_Habits_pre, Smoker_pre, Family_Heart_Disease_pre,
                                                                  Diabetes_pre,BMI_pre,High_Blood_Pressure_pre,Low_HDL_Cholesterol_pre, High_LDL_Cholesterol_pre,Alcohol_Consumption_pre, 
                                                                  Stress_pre, Sleep_Hours_pre,Sugar_Consumption_pre, Triglyceride_Level_pre, Fasting_Blood_Sugar_pre,CRP_Level_pre, 
                                                                  Homocysteine_Level_pre]])

    # Load Model


        
    model = joblib.load('Catboost_Model.pkl')

    result = model.predict(predicted_data)[0]

    button = colA.button('Predict HD Probability')

    if button == True:

        if result == 1:
            colA.error('HD Probabily Positive')
            colA.write(model.predict_proba(predicted_data).round(3)[0][1] *100)

        else:
            colA.success('HD Nigative')
            colA.write(model.predict_proba(predicted_data).round(3)[0][1] *100)


    # Display Important Features Table
    colB.subheader("📝 Important Features")
    colB.table(importance_df)


