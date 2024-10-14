import streamlit as st
import pandas as pd
import pickle


with open('C:/Users/Shehryar Hasan/Titanic_Survival_Prediction/model/titanic_model.pkl', 'rb') as f:
    model = pickle.load(f)



st.title('Titanic Survival Prediction App')
st.write('Enter passenger details to predict survival.')

sex = st.selectbox('Sex', ['Male', 'Female'])
age = st.slider('Age', 1, 100, 30)
pclass = st.selectbox('Pclass', [1, 2, 3])
sibsp = st.number_input('Number of Siblings/Spouses Aboard', min_value=0, max_value=10, value=0)
parch = st.number_input('Number of Parents/Children Aboard', min_value=0, max_value=10, value=0)
fare = st.number_input('Fare', min_value=0.0, value=10.0)
embarked = st.selectbox('Embarked', ['S', 'C', 'Q'])


family_size = sibsp + parch + 1


input_data = pd.DataFrame({
    'Pclass': [pclass],
    'Sex': [1 if sex == 'Female' else 0],
    'Age': [age],
    'SibSp': [sibsp],
    'Parch': [parch],
    'Fare': [fare],
    'Embarked': [0 if embarked == 'S' else 1 if embarked == 'C' else 2],
    'FamilySize': [family_size]  # Add FamilySize feature
})


if st.button('Predict'):
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.success('The passenger would have survived.')
    else:
        st.error('The passenger would not have survived.')
