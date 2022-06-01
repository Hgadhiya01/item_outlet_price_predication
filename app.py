from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd


df = pd.read_csv("train.csv")

num_cols = df.select_dtypes(["int64","float64"]).keys()
cat_cols = df.select_dtypes(["object"]).keys()


for var in num_cols:
    df[var].fillna(df[var].mean(),inplace = True)

for var in cat_cols:
    df[var].fillna(df[var].mode()[0],inplace = True)

df1 = df.copy()

from sklearn.preprocessing import LabelEncoder
lr = LabelEncoder()
for var in cat_cols:
    lr.fit(df1[var])
    df1[var]= lr.transform(df1[var])


df_cols = df.drop("Item_MRP",axis=1).keys()
di = {}
for var in cat_cols: #item identifier
    for var1,var2 in zip(df[var], df1[var]): #fda15, 156
        di[var1]=var2


model = joblib.load("Item_outlet_sales_prediction.pkl")

def item_outlet_price_prediction(model, Item_Identifier, Item_Weight, Item_Fat_Content, Item_Visibility,
                                Item_Type, Outlet_Identifier,Outlet_Establishment_Year, Outlet_Size, 
                            Outlet_Location_Type, Outlet_Type, Item_Outlet_Sales):
    x=np.zeros(11)
    x[1]= Item_Weight
    x[3]=Item_Visibility
    x[6]=Outlet_Establishment_Year
    x[10]=Item_Outlet_Sales
    x[0] = di[Item_Identifier]
    x[2] = di[Item_Fat_Content]
    x[4] = di[Item_Type]
    x[5] = di[Outlet_Identifier]
    x[7] = di[Outlet_Size]
    x[8] = di[Outlet_Location_Type]
    x[9] = di[Outlet_Type]
    
    
    return model.predict([x])[0]

app=Flask(__name__)
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    Item_Identifier = request.form["Item_Identifier"]
    Item_Weight = request.form["Item_Weight"]
    Item_Fat_Content = request.form["Item_Fat_Content"]
    Item_Visibility = request.form["Item_Visibility"]
    Item_Type = request.form["Item_Type"]
    Outlet_Identifier = request.form["Outlet_Identifier"]
    Outlet_Establishment_Year = request.form["Outlet_Establishment_Year"]
    Outlet_Size = request.form["Outlet_Size"]
    Outlet_Location_Type = request.form["Outlet_Location_Type"]
    Outlet_Type = request.form["Outlet_Type"]
    Item_Outlet_Sales = request.form["Item_Outlet_Sales"]

    
    predicated_price1 =item_outlet_price_prediction(model, Item_Identifier, Item_Weight, Item_Fat_Content, Item_Visibility,
                                Item_Type, Outlet_Identifier,Outlet_Establishment_Year, Outlet_Size, 
                            Outlet_Location_Type, Outlet_Type, Item_Outlet_Sales)
    predicated_price = round(predicated_price1, 2)

    return render_template("index.html", prediction_text="Predicated price of Item-outlet sales is {} RS".format(predicated_price))


if __name__ == "__main__":
    app.run()    
    
