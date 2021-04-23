# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 18:26:33 2021

@author: Praveen Bhargav
"""

from pywebio.input import *
from pywebio.output import *
from pywebio import start_server
from pywebio.exceptions import SessionClosedException
from pywebio.platform.flask import webio_view
from flask import Flask, render_template, request
from pywebio import STATIC_PATH
import pandas  as pd
import pickle as pk
import warnings
import argparse
warnings.filterwarnings("ignore")
from pycaret.regression import *
from pycaret.regression import load_model
from datetime import date

rf_model=load_model("saved_rfmodel")

app = Flask(__name__)
def data_gathering():
    put_markdown("## Second Hand Car Price Prediction('Using Pywebio')",lstrip=True)
    
    web_inputs= input_group(
        "Enter the following Information",
        [
            input("Present Price:",type=FLOAT,name="present_price"),
            input("Kilometers Driven:", type=FLOAT,name="km_driven" ),
            radio("Fuel_Type:",options=[("Petrol","Petrol"),("Diesel","Diesel")],name="fuel_type"),
            radio("Seller_Type:",name="seller_type",options=[("Dealer","Dealer"),("Individual","Individual")]),
            radio("Transmission",name="transmission",options=[(i,i) for i in ("Manual","Automatic")]),
            input("No.Of Owners:",name="owners",type=NUMBER),
            input("Year:",name="year",type=NUMBER)
            ])
    today_date= date.today()
    cur_year=today_date.year
    years_used= int(cur_year - web_inputs["year"])
#     put_text("year: %d" %(years_used))
    
    df= pd.DataFrame(data=[[web_inputs[i] for i in ["present_price","km_driven","fuel_type","seller_type","transmission","owners"]]],
                     columns=["Present_Price","Kms_Driven","Fuel_Type","Seller_Type","Transmission","Owner"],index=[0])
    
    df["Years_Used"]= years_used
    df["Years_Used"] = df["Years_Used"].astype(int)
#     put_text("year: %d" %(int(df["Years_Used"])))

    pred= predict_model(rf_model,data=df)
    pred= float(pred["Label"])
    put_markdown("### Predicted Selling Price:%.2f L" %(pred))
    
    put_text("\n\n\n\n\n\n\n\n\n\t\t\t©MOTURU PRAVEEN BHARGAV 2021")
    
    
app.add_url_rule('/predict', 'webio_view', webio_view(data_gathering),
            methods=['GET', 'POST', 'OPTIONS'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--port", type=int, default=8080)
    args = parser.parse_args()

    start_server(data_gathering, port=args.port)

# app.run(host='localhost', port=80)
    