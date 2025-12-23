import streamlit as st
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

#page config 

st.set_page_config("Linear regression ",layout="centered")

def load_css(file):
    with open(file) as f:
        st.markdown(f"<style> {f.read()} </style>",unsafe_allow_html=True)

load_css("style.css")

# title
st.markdown("""
        <div class="card">
        <h1>Linear regression </h1>
        <p> predict <b> tip amount </b> total bill </b> using linear regression </p>
            """,unsafe_allow_html=True)

# load data
@st.cache_data
def load_data():
    return sns.load_dataset("tips")
df=load_data()

#dataset preview

st.markdown('<div class= "card">',unsafe_allow_html=True)
st.subheader("dataset preview")
st.dataframe(df.head())
st.markdown('<div>',unsafe_allow_html=True)

# prepare the data
x,y=df[["total_bill"]],df["tip"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)

scalar=StandardScaler()

x_train=scalar.fit_transform(x_train)
x_test=scalar.transform(x_test)

# train model
model=LinearRegression()
model.fit(x_train,y_train)

y_pred=model.predict(x_test)

# metrics

mae=mean_absolute_error(y_test,y_pred)

rmse=np.sqrt(mae)

r2=r2_score(y_test,y_pred)

adj_r2=1-(1-r2)*(len(y_test)-1)/(len(y_test)-2)

# visualization

st.markdown('<div class="card">',unsafe_allow_html=True)
st.subheader("total bill vs tip")

fig,ax=plt.subplots()
ax.scatter(df["total_bill"],df["tip"],alpha=0.6)
ax.plot(df["total_bill"],model.predict(scalar.transform(x)),color="red")
ax.set_xlabel("total bill")
ax.set_ylabel("tip")

st.pyplot(fig)
st.markdown('</div>',unsafe_allow_html=True)

# performance

st.markdown('<div class="card>',unsafe_allow_html=True)
st.subheader("model performance")

c1,c2=st.columns(2)
c1.metric("MAE",f"{mae:.2f}")
c2.metric("RMAE",f"{rmse:.2f}")

c3,c4=st.columns(2)
c3.metric("R2",f"{r2:.2f}")
c4.metric("adj R2",f"{adj_r2:.2f}")

st.markdown('</div>',unsafe_allow_html=True)

# m & c
st.markdown(f"""
            <div class="card">
            <h3> model intercept & co efficient <h3>
            <p> <b> co efficient : </b> {model.coef_[0]:.3f}<br>
            <h> intercept : </h> {model.intercept_:.3f}</p>
            </div>
            """,unsafe_allow_html=True)

# prediction 

st.markdown('<div class="card">',unsafe_allow_html=True)
st.subheader("predict tip amount ")

bill=st.slider("total bill : $",float(df.total_bill.min()),float(df.total_bill.max()),30.0)
tip=model.predict(scalar.transform([[bill]]))[0]

st.markdown(f'<div class="prediction-box"> predict tip : $ {tip:.2f} </div>',unsafe_allow_html=True)
st.markdown('<div> ',unsafe_allow_html=True)