from django.shortcuts import render

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
import pandas as pd



# Create your views here.

def predict_price(request):
    
    return render(request, "prediction.html")
