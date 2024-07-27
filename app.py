# import numpy as np
# from flask import Flask, request, jsonify, render_template
# import pickle

# # Create flask app
# flask_app = Flask(__name__)
# model = pickle.load(open("model.pkl", "rb"))

# @flask_app.route("/")
# def Home():
#     return render_template("index.html")

# @flask_app.route("/predict", methods = ["POST"])
# def predict():
#     float_features = [float(x) for x in request.form.values()]
#     features = [np.array(float_features)]
#     prediction = model.predict(features)
#     return render_template("index.html", prediction_text = "The flower species is {}".format(prediction))

# if __name__ == "__main__":
#     flask_app.run(debug=True)

from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import numpy as np
import pickle
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request

# Create FastAPI app
app = FastAPI()
model = pickle.load(open("model.pkl", "rb"))
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, Sepal_Length: float = Form(...), Sepal_Width: float = Form(...), Petal_Length: float = Form(...), Petal_Width: float = Form(...)):
    features = np.array([[Sepal_Length, Sepal_Width, Petal_Length, Petal_Width]])
    prediction = model.predict(features)[0]
    print(prediction)
    return templates.TemplateResponse("index.html", {"request": request, "prediction_text": f"The flower species is {prediction}"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
