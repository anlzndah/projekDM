import streamlit as st 
import pandas as pd 
from PIL import Image
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB

st.write("""
# Web Apps - Klasifikasi Bunga Mawar
Aplikasi berbasis web untuk mengklasifikasi jeni-jenis bunga mawar
"""
)

img=Image.open('roses.jpg')
st.image(img, use_column_width=False)

st.sidebar.header('Parameter Inputan')

def input_user():
    panjang_sepal=st.sidebar.slider('Panjang Sepal', 4.3, 7.9, 5.4)
    lebar_sepal=st.sidebar.slider('Lebar Sepal', 2.0, 4.4, 3.4)
    panjang_petal=st.sidebar.slider('Panjang Petal', 1.0, 6.9, 1.3)
    lebar_petal=st.sidebar.slider('Lebar Petal', 0.1, 2.5, 0.2)
    data={'panjang sepal': panjang_sepal,
            'lebar sepal' : lebar_sepal,
            'panjang petal' : panjang_petal,
            'lebar petal' : lebar_petal}
    fitur=pd.DataFrame(data, index=[0])
    return fitur 
df=input_user()

st.subheader('Parameter Inputan')
st.write(df)

rose=datasets.load_iris()
X=rose.data
Y=rose.target

model=GaussianNB()
model.fit(X, Y)

prediksi=model.predict(df)
prediksi_proba=model.predict_proba(df)

st.subheader('Label kelas dan nomor indeks yang sesuai')
st.write(rose.target_names)

st.subheader('Prediksi (Hasil Klasifikasi)')
st.write(rose.target_names[prediksi])

st.subheader('Probabilitas Hasil Prediksi (Klasifikasi)')
st.write(prediksi_proba)
