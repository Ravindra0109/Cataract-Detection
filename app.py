import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask,render_template,request

app=Flask(__name__) # initializing the flask application

model=load_model(r'C:\Users\deeks\OneDrive\Desktop\cataract\cataract detection\vggsig.h5')
@app.route('/')
def index():
    return render_template("index.html")
@app.route('/predict',methods=['GET','POST'])
def upload():
    if request.method=='POST':
        f=request.files['image']
        basepath=os.path.dirname(__file__)
        filepath=os.path.join(basepath,'uploads',f.filename)
        f.save(filepath)

        img=image.load_img(filepath,target_size=(224,224))
        x=image.img_to_array(img)
        print("Shape before expanding:",x.shape)
        x=np.expand_dims(x,axis=0)
        print("shape after expanding: ",x.shape)
        pred=np.argmax(model.predict(x))
        index =['cataract','normal']
       
        text="The Classified image is " +str(index[pred])
    return text

if __name__=='__main__':
    app.run(debug=True, port=8080) #run the flask application