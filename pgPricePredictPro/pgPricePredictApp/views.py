import os

import pandas as pd

from django.conf import settings
from django.shortcuts import render

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline


def predict_price(request):

    # Load dataset
    df = pd.read_csv("pg_price_prediction_dataset_1850.csv")

    # Features and target
    X = df.drop("rent", axis=1)
    y = df["rent"]

    # Categorical columns
    categorical_features = [
        "city",
        "area",
        "room_type",
        "food",
        "wifi",
        "ac",
        "attached_bathroom",
        "furnished",
        "parking",
        "cctv",
        "power_backup"
    ]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42
    )

    # Preprocessor
    process = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features
            )
        ],
        remainder="passthrough"
    )

    # Model
    model = Pipeline([
        ("preprocessor", process),
        ("regressor", DecisionTreeRegressor(random_state=42))
    ])

    # Train model
    model.fit(X_train, y_train)

    # POST request
    if request.method == "POST":

        city = request.POST.get("city")
        area = request.POST.get("area")
        room_type = request.POST.get("room_type")

        sharing = int(request.POST.get("sharing"))
        room_size_sqft = float(request.POST.get("room_size_sqft"))

        food = request.POST.get("food")
        wifi = request.POST.get("wifi")
        ac = request.POST.get("ac")
        attached_bathroom = request.POST.get("attached_bathroom")
        furnished = request.POST.get("furnished")
        parking = request.POST.get("parking")
        cctv = request.POST.get("cctv")
        power_backup = request.POST.get("power_backup")

        # Input data
        input_data = pd.DataFrame({
            "city": [city],
            "area": [area],
            "room_type": [room_type],
            "sharing": [sharing],
            "room_size_sqft": [room_size_sqft],
            "food": [food],
            "wifi": [wifi],
            "ac": [ac],
            "attached_bathroom": [attached_bathroom],
            "furnished": [furnished],
            "parking": [parking],
            "cctv": [cctv],
            "power_backup": [power_backup]
        })

        # Prediction
        result = model.predict(input_data)

        return render(
            request,
            "prediction.html",
            {
                "result": result[0]
            }
        )
        print("""================================================================================================================================================================================================
              =========================================
              ==========================================
              ===================================""")

    return render(request, "prediction.html")