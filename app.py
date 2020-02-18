#Test application working fine

#from load import *
from flask import Flask, render_template, request
from flask_uploads import UploadSet, configure_uploads, IMAGES
from scipy.misc import imsave, imread, imresize
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions

import numpy as np
import keras.models
from keras.models import model_from_json
from scipy.misc import imread, imresize,imshow
import tensorflow as tf

global model, graph

json_file = open('model.json','r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
#load woeights into new model
model.load_weights("model.h5")
print("Loaded Model from disk")

#compile and evaluate loaded model
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
#loss,accuracy = model.evaluate(X_test,y_test)
#print('loss:', loss)
#print('accuracy:', accuracy)
graph = tf.get_default_graph()

#return loaded_model,graph


# for matrix math
import numpy as np
from werkzeug import secure_filename
# for importing our keras model
import keras.models
# for regular expressions, saves time dealing with string data
import re
# system level operations (like loading files)
import sys
# for reading operating system data
import os
# tell our app where our saved model is
#sys.path.append(os.path.abspath("./model"))
# initalize our flask app
app = Flask(__name__)
# global vars for easy reusability

# initialize these variables
#model, graph = init()

photos = UploadSet('photos', IMAGES)

app.config['UPLOADED_PHOTOS_DEST'] = '.'
configure_uploads(app, photos)


@app.route('/')
def index():
    # initModel()
    # render out pre-built HTML file right on the index page
    return render_template("index.html")


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST' and 'photo' in request.files:
        filename = photos.save(request.files['photo'])
        os.rename('./'+filename, './'+'output.png')

        print ("debug")
        # read the image into memory
        img = image.load_img('./output.png', target_size=(224, 224))
        # compute a bit-wise inversion so black becomes white and vice versa
        x = image.img_to_array(img)
        # make it the right size
        x = np.expand_dims(x, axis=0)
        # imshow(x)
        # convert to a 4D tensor to feed into our model
        x = preprocess_input(x)
        print ("debug2")
        # in our computation graph
        with graph.as_default():
            # perform the prediction
            out = model.predict(x)
            u = decode_predictions(out, top=3)[0]
            s1 = u[0][1]
            s2 = u[0][2]*100
            s3 = u[1][1]
            s4 = u[1][2]*100
            s5 = u[2][1]
            s6 = u[2][2]*100
            print(s1, s2, s3)
            print ("debug3")
            # convert the response to a string
            return render_template("index2.html", s1=s1, s2=s2, s3=s3, s4=s4, s5=s5, s6=s6)


if __name__ == "__main__":
    # decide what port to run the app in
    #port = int(os.environ.get('PORT', 5000))
    # run the app locally on the givn port
    #app.run(host='0.0.0.0', port=port)
    app.run(threaded=True, port=5000)
    # optional if we want to run in debugging mode
    app.run(debug=True)
