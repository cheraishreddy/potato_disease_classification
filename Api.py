from fastapi import  FastAPI,UploadFile,File
import uvicorn
import tensorflow as tf
import numpy as np
import PIL as pl
from io import BytesIO
from fastapi.middleware.cors import CORSMiddleware
app=FastAPI()
model=tf.keras.models.load_model('.\modelk')
origins=[
    "http://127.0.0.1:3000",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)
@app.post('/hello')
async def fun(file:UploadFile= File(...)):
    data=await file.read()
    image=np.array(pl.Image.open(BytesIO(data)))
    dat=model.predict(np.array([image]))
    o=np.argmax(dat)
    r=np.max(dat)
    return {'data':str(o) ,'conf':str(r*100)}
if __name__== '__main__' :
    uvicorn.run("Api:app",port=8080)

