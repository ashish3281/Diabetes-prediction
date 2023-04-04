import numpy as np
import pickle
import streamlit as st

loaded_model = pickle.load(open('C:/Users/HP/OneDrive/Desktop/ML projects/Diabetes_model.sav', 'rb'))

def Diabetes_predicton(input_data):


# changing the input_data to numpy array
   input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
   input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)



   prediction = loaded_model.predict(input_data_reshaped)
   print(prediction)

   if (prediction[0] == 0):
     return 'The person is not diabetic'
   else:
     return 'The person is diabetic'
  
  
def main():
      
      st.title('Diabetes Prediction Model')
      
      Pregnancies=st.text_input("Number of pregancies")
      Glucose=st.text_input("Glucose level")
      BloodPressur=st.text_input("Blood Pressure Value")
      SkinThickness=st.text_input("Skin Thickness Value")
      Insulin=st.text_input("Insulin Value")
      BMI=st.text_input("BMI Value")
      DiabetesPedigreeFunction=st.text_input("Diabetes Pedigree Function Value")
      Age=st.text_input("Age of the Person")
      
      
      diagnosis=''
      
      if st.button('Diabetes Test Result'):
          diagnosis=Diabetes_predicton([Pregnancies, Glucose, BloodPressur, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])
          
      st.success(diagnosis)
      
      
if __name__=='__main__':
    main()
      
      
      
      
