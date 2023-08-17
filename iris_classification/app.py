import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import pickle 

model=pickle.load(open('model.pkl', 'rb'))
data=pd.read_csv("Iris.csv")



def iris_prediction(input_data):
  y_new_pred=model.predict(pd.DataFrame([input_data]))
  if(y_new_pred==0):
    return "Iris sentos"
  elif (y_new_pred==1):
    return "Iris Virsicolor"
  elif(y_new_pred==2):
    return "Iris Virginica"
  



def main():
  st.title("Iris classification application")

  SepalLengthCm=data['SepalLengthCm']
  SepalWidthCm=data['SepalWidthCm']
  PetalLengthCm=data['PetalLengthCm']
  PetalWidthCm=data['PetalWidthCm']

  SepalLengthCm=st.number_input("Enter Sepel Length")
  SepalWidthCm=st.number_input("Enter Sepel width")
  PetalLengthCm=st.number_input("Enter petal length")
  PetalWidthCm=st.number_input("Enter petal width")


  prediction=''

  if st.button("Predict"):
    prediction=iris_prediction([SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm])
  
    

  st.success(prediction)
  if prediction=='Iris sentos': 
      image = Image.open('iris_sentos.jpg')
      st.image(image, caption='Iris Sentos')
  elif prediction=="Iris Virginica":
      image = Image.open('Iris-virginica.jpg')
      st.image(image, caption='Iris Verginica')

  elif prediction=="Iris Virsicolor":
    image=Image.open('Iris-versicolor.jpg')
    st.image(image,caption="iris versicolor")





if __name__=='__main__':
  main()
  
