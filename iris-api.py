from fastapi import FastAPI, Query
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
import uvicorn
import pandas as pd
import pickle
import base64

app = FastAPI(title="IRIS API")

@app.get("/")
def homepage():
    html = """
        <html>
        <head><title>IRIS API</title></head>
        <body>
            <h1>Welcome to the IRIS API</h1>
            <p>For more information, visit
            <a href="https://google.com">google.com</a>
        </body>
        </html>
    """
    return HTMLResponse(content=html, status_code=200)

@app.get("/iris")
def get_classification(petal_length: float=Query(description="Petal Length", default=1.4, ge=1, le=6.9),
                       petal_width: float=Query(description="Petal Width", default=0.2, ge=0.1, le=2.5),
                       sepal_length: float=Query(description="Sepal Length", default=4.9, ge=4.3, le=7.9),
                       sepal_width: float=Query(description="Sepal Width", default=3.0, ge=2, le=4.4)):
    
    column_names = ["sepal length (cm)", "sepal width (cm)",
                    "petal length (cm)", "petal width (cm)"]
    df = pd.DataFrame(data=[[sepal_length, sepal_width,
                             petal_length, petal_width]],
                      columns=column_names)

    pipe = pickle.load(open("iris-pipe.pkl", "rb"))
    prediction = pipe.predict(df)
    class_name = prediction[0]

    file_name = f"data/{class_name}.jpg"
    with open(file_name, "rb") as f:
        binary_img = f.read()
        
    encoded_img = base64.b64encode(binary_img)
    return {"class": class_name,
            "image": encoded_img}
    
if __name__ == "__main__":
    uvicorn.run("iris-api:app",
                host="0.0.0.0",
                port=8000,
                reload=True)