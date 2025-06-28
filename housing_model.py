import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
import joblib



df=pd.read_csv("D:/Data_Science/Data/housing_new.csv")
df1=df

#print(df['ocean_proximity'])

# for i in df.head():
#     print(i)

#print(df['ocean_proximity'].unique())
dummies_ocean_proximity=pd.get_dummies(df.ocean_proximity,dtype=int)
#print(dummies_ocean_proximity)

#ocean_p_encoder.fit(df['ocean_proximity'].unique())

df=df.drop(columns='ocean_proximity',axis='columns')
#print(df)
df_merged=pd.concat([df,dummies_ocean_proximity],axis='columns')

#print(df_merged["median_house_value"])
# for i in df_merged.head():
#     print(i)

x=df_merged.drop(["median_house_value"],axis="columns")
y=df_merged["median_house_value"]

# for j in x.head():
#     print(j)


imputer=SimpleImputer(strategy='mean')
model=make_pipeline(imputer,LinearRegression())
model.fit(x,y)

# for j in y:
#     if j=='NaN':
#         print(j)

joblib.dump(model,"housing_price_prediction.pkl")
