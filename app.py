# My two categories
X = "Benign"
Y = "Malignant"

# Two example images for the website, they are in the static directory next 
# where this file is and must match the filenames here
sampleX='../static/benign.png'
sampleY='../static/malignant.png'

# Where I will keep user uploads
UPLOAD_FOLDER = 'C:/Users/Waseem Ahmad Qureshi/Downloads/static/uploads'
# Allowed files
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
# Machine Learning Model Filename
CNN_MODEL_FILENAME = 'C:/Users/Waseem Ahmad Qureshi/Downloads/Data/Models/cnn.h5'
TF_MODEL_FILENAME = 'C:/Users/Waseem Ahmad Qureshi/Downloads/Data/Models/vgg16.h5'
#Load operation system library
import os

#website libraries
from flask import render_template
from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename

#Load math library
import numpy as np

#Load machine learning libraries
from tensorflow.keras.preprocessing import image
from keras.models import load_model
from keras.backend import set_session
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
# Create the website object
app = Flask(__name__)

def load_model_from_file():
    #Set up the machine learning session
    mySession = tf.compat.v1.Session()
    set_session(mySession)
    myModel = load_model(CNN_MODEL_FILENAME)
    myGraph = tf.compat.v1.get_default_graph()
    myModel2 = load_model(TF_MODEL_FILENAME)
    myGraph2 = tf.compat.v1.get_default_graph()
    return (mySession,myModel,myGraph,myModel2,myGraph2)

#Try to allow only images
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

#Define the view for the top level page
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    #Initial webpage load
    if request.method == 'GET' :
        return render_template('index.html',myX=X,myY=Y,mySampleX=sampleX,mySampleY=sampleY)
    else: # if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser may also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        # If it doesn't look like an image file
        if not allowed_file(file.filename):
            flash('I only accept files of type'+str(ALLOWED_EXTENSIONS))
            return redirect(request.url)
        #When the user uploads a file with good parameters
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('uploaded_file', filename=filename))

    
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    test_image = image.load_img(UPLOAD_FOLDER+"/"+filename,target_size=(50,50))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)

    mySession = app.config['SESSION']
    myModel = app.config['MODEL']
    myGraph = app.config['GRAPH']
    with myGraph.as_default():
        set_session(mySession)
        result = myModel.predict(test_image)
        image_src = '../static/uploads/' +filename
        if result[0] < 0.5 :
            answer = "<div class='col text-center'><img width='75' height='75' src='"+image_src+"' class='img-thumbnail' /><h5>Diagnosis: <span style='color:#66CC66'>"+X+" </span></h5></div><div class='col'></div><div class='w-100'></div>"     
        else:
            answer = "<div class='col'></div><div class='col text-center'><img width='75' height='75' src='"+image_src+"' class='img-thumbnail' /><h5>Diagnosis: <span style='color:#FF0000'>"+Y+"</span>  </h5></div><div class='w-100'></div>"     
        
        return render_template('index.html',myX=X,myY=Y,mySampleX=sampleX,mySampleY=sampleY,results=answer)

    myModel2 = app.config['MODEL2']
    myGraph2 = app.config['GRAPH2']
    with myGraph2.as_default():
        set_session(mySession)
        result = myModel2.predict(test_image)
        image_src = '../static/uploads/' +filename
        if result[0] < 0.5 :
            answer = "<div class='col text-center'><img width='75' height='75' src='"+image_src+"' class='img-thumbnail' /><h5>Diagnosis: <span style='color:#66CC66'>"+X+" </span></h5></div><div class='col'></div><div class='w-100'></div>"     
        else:
            answer = "<div class='col'></div><div class='col text-center'><img width='75' height='75' src='"+image_src+"' class='img-thumbnail' /><h5>Diagnosis: <span style='color:#FF0000'>"+Y+"</span>  </h5></div><div class='w-100'></div>"    
        
        return render_template('index.html',myX=X,myY=Y,mySampleX=sampleX,mySampleY=sampleY,results=answer)


def main():
    (mySession,myModel,myGraph,myModel2,myGraph2) = load_model_from_file()
    
    app.config['SECRET_KEY'] = 'super secret key'
    
    app.config['SESSION'] = mySession
    app.config['MODEL'] = myModel
    app.config['GRAPH'] = myGraph
    app.config['MODEL2'] = myModel2
    app.config['GRAPH2'] = myGraph2
    
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 #16MB upload limit
    app.run()



#Launch everything
main()
