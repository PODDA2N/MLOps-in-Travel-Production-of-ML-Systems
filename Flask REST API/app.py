#!/usr/bin/env python
# coding: utf-8

# In[1]:


from flask import Flask
from flask import request
from flask import jsonify

import pandas as pd
import joblib


# #### Load Model

# In[27]:


model = joblib.load("E:\Data Analyst Coding\Machine Learning Project Internship\Travel_MLOps_Project\Models/flight_model.pkl")

scaler = joblib.load("E:\Data Analyst Coding\Machine Learning Project Internship\Travel_MLOps_Project\Models/scaler.pkl")

encoders = joblib.load("E:\Data Analyst Coding\Machine Learning Project Internship\Travel_MLOps_Project\Models/label_encoders.pkl")


# #### Create Flask App

# In[28]:


app = Flask(__name__)


# In[ ]:


@app.route("/predict_flight", methods=["POST"])
def predict_flight():

    try:

        # Read JSON data
        data = request.get_json()

        # Convert JSON to DataFrame
        df = pd.DataFrame([data])

        # --------------------------
        # Date Feature Engineering
        # --------------------------
        df["date"] = pd.to_datetime(df["date"])

        df["TravelMonth"] = df["date"].dt.month
        df["TravelDay"] = df["date"].dt.day
        df["TravelWeekday"] = df["date"].dt.weekday
        df["Weekend"] = (df["TravelWeekday"] >= 5).astype(int)

        df.drop("date", axis=1, inplace=True)

        # --------------------------
        # Encode Categorical Features
        # --------------------------
        df["from"] = encoders["from"].transform(df["from"])
        df["to"] = encoders["to"].transform(df["to"])
        df["flightType"] = encoders["flightType"].transform(df["flightType"])
        df["agency"] = encoders["agency"].transform(df["agency"])

        # --------------------------
        # Normalize Numerical Features
        # --------------------------
        df[["time", "distance"]] = scaler.transform(
            df[["time", "distance"]]
        )

        # --------------------------
        # Prediction
        # --------------------------
        prediction = model.predict(df)

        # --------------------------
        # Return JSON
        # --------------------------
        return jsonify({
            "Predicted Flight Price": round(float(prediction[0]), 2)
        })

    except Exception as e:

        return jsonify({
            "Error": str(e)
        })

# ==========================
# Run Flask
# ==========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




