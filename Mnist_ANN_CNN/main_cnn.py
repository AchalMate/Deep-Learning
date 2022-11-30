from flask import Flask, render_template, request
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model


app = Flask(__name__, template_folder='templates')


def init():
    global model, graph
    model = load_model('mnist_seq.h5')
    graph = tf.compat.v1.get_default_graph()


@app.route('/')
def upload_file():
    return render_template('cnn_home.html')


@app.route('/', methods=['POST','GET'])
def upload_image_file():
    if request.method == 'POST':
        img = Image.open(request.files['file_mnist'].stream).convert("L")
        img = img.resize((28, 28))
        im2arr = np.array(img)
        im2arr = im2arr.reshape(1, 28,28,1)
        with graph.as_default():
            model = load_model('mnist_cnn.h5')
            predict_x = model.predict(im2arr)
            classes_x = np.argmax(predict_x, axis=1)


        print("Predicted Number: " + str(classes_x[0]))
        return render_template('cnn_home.html', ans=str(classes_x[0]), img=img)


if __name__ == '__main__':
    print("* Loading ")
    init()
    app.run(debug=True)
