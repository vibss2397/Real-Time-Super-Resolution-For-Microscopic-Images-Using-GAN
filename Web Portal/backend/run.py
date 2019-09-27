from flask import Flask, request, render_template, send_file, json
from PIL import Image
import requests
import base64
from io import BytesIO
import json
from keras.models import load_model
import numpy as np
from PIL import Image
import tensorflow as tf
#from flask_socketio import SocketIO, send, emit
#from flask_cors import CORS

app = Flask(__name__)
model = None
model2 = None
model3 = None
model4 = None
model5 = None
model6 = None
#app.config['SECRET_KEY'] = 'secret!' 
#io = SocketIO(app)
#clients = []
#CORS(app)
port_mobile = 'http://192.168.43.250'
port_wifi = 'http://192.168.1.9'
def load2():
    global model, model2, model3, model4, model5, model6
    model = load_model('model/models/gen_down.hdf5')
    model._make_predict_function()
    model2 = load_model('model/models/gen_64block_down512.hdf5')
    model2._make_predict_function()
    model3 = load_model('model/models/original_srgan.hdf5', custom_objects={"tf": tf})
    model3._make_predict_function()
    model4 = load_model('model/models/subpixel+improved_disc.hdf5', custom_objects={"tf": tf})
    model4._make_predict_function()
    model5 = load_model('model/models/inter+improved_disc.hdf5', custom_objects={"tf": tf})
    model5._make_predict_function()
    model6 = load_model('model/models/inter+improved_disc-bn.hdf5', custom_objects={"tf": tf})
    model6._make_predict_function()


def normalize(input_data):
    return (input_data.astype(np.float32) - 127.5)/127.5 

def denormalize(input_data):
    input_data = (input_data + 1) * 127.5
    return np.uint8(input_data)

def serve_pil_image(pil_img):
    img_io = BytesIO()
    pil_img.save(img_io, 'JPEG', quality=70)
    img_io.seek(0)
    return send_file(img_io, mimetype='image/jpeg')

@app.route('/')
def hello():
    im=Image.open(requests.get('https://ibidi.com/img/cms/applications/technical_aspects/microscopy_techniques/TA_WFM_Rat1.jpg', stream=True).raw)
    im = im.resize((500, 500))
    return serve_pil_image(im)


@app.route('/app')
def hello3():
    return render_template('app.html')

@app.route('/send_request', methods=['POST'])
def hello2():
    data = json.loads(request.get_data().decode('utf-8'))
    img_data = requests.get(data['domain']).content
    with open('./static/img/aaa.png', 'wb') as handler:
        handler.write(img_data)
    im = Image.open('./static/img/aaa.png')
    im.save('./static/img/recieve.png', quality=100, subsampling=0)
    print(int(data['coordinate']['x1']))
    im = im.crop((int(data['coordinate']['x1']), int(data['coordinate']['y1']), int(data['coordinate']['x2']), int(data['coordinate']['y2'])))
    im.save('./static/img/cropped.png')
    buffered = BytesIO()
    im.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue())
    return img_str

@app.route('/upsample', methods=['POST'])
def upsample():
    global model, model2, model3, model4, model5, model6
    import time
    img_dict = {}
    str2 = ''
    data = json.loads(request.get_data().decode('utf-8'))
    im = Image.open(BytesIO(base64.b64decode(data['image'])))
    im = im.resize((64, 64), Image.ANTIALIAS)    
    arr = np.array(im).reshape(1, 64, 64, 3)
    arr2 = (np.float32(arr) - 127.5)/127.5
    t1 = time.time()
    predict = model2.predict(arr2)
    print(time.time()-t1)
    predict = predict.reshape(256, 256, 3)
    prediction2 = np.uint8((predict + 1) * 127.5)
    img2 = Image.fromarray(prediction2)
    img2.save('./static/img/5.png')
    buffered = BytesIO()
    img2.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue())
    # img_dict['sr'] = str(img_str)
    str2+='    '
    str2+=str(img_str)

    im2 = Image.open(BytesIO(base64.b64decode(data['image'])))
    im2 = im2.resize((64, 64), Image.ANTIALIAS)
    arr = np.array(im2).reshape(1, 64, 64, 3)
    arr2 = (np.float32(arr) - 127.5)/127.5
    predict = model.predict(arr2).reshape(256, 256, 3)
    prediction2 = np.uint8((predict + 1) * 127.5)
    img3 = Image.fromarray(prediction2)
    img3.save('./static/img/6.png')
    buffered = BytesIO()
    img3.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue())
    # img_dict['sr'] = str(img_str)
    str2+='    '
    str2+=str(img_str)

    im3 = Image.open(BytesIO(base64.b64decode(data['image'])))
    im3 = im3.resize((64, 64), Image.ANTIALIAS)
    arr = np.array(im3).reshape(1, 64, 64, 3)
    arr2 = (np.float32(arr) - 127.5)/127.5
    predict = model3.predict(arr2).reshape(256, 256, 3)
    prediction2 = np.uint8((predict + 1) * 127.5)
    img4 = Image.fromarray(prediction2)
    img4.save('./static/img/7.png')
    buffered = BytesIO()
    img4.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue())
    # img_dict['sr'] = str(img_str)
    str2+='    '
    str2+=str(img_str)

    im4 = Image.open(BytesIO(base64.b64decode(data['image'])))
    im4 = im4.resize((64, 64), Image.ANTIALIAS)
    arr = np.array(im4).reshape(1, 64, 64, 3)
    arr2 = (np.float32(arr) - 127.5)/127.5
    predict = model4.predict(arr2).reshape(256, 256, 3)
    prediction2 = np.uint8((predict + 1) * 127.5)
    img5 = Image.fromarray(prediction2)
    img5.save('./static/img/8.png')
    buffered = BytesIO()
    img5.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue())
    # img_dict['sr'] = str(img_str)
    str2+='    '
    str2+=str(img_str)

    im5 = Image.open(BytesIO(base64.b64decode(data['image'])))
    im5 = im5.resize((64, 64), Image.ANTIALIAS)
    arr = np.array(im5).reshape(1, 64, 64, 3)
    arr2 = (np.float32(arr) - 127.5)/127.5
    predict = model5.predict(arr2).reshape(256, 256, 3)
    prediction2 = np.uint8((predict + 1) * 127.5)
    img6 = Image.fromarray(prediction2)
    img6.save('./static/img/9.png')
    buffered = BytesIO()
    img6.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue())
    # img_dict['sr'] = str(img_str)
    str2+='    '
    str2+=str(img_str)

    im8 = Image.open(BytesIO(base64.b64decode(data['image'])))
    im8 = im8.resize((64, 64), Image.ANTIALIAS)
    arr = np.array(im8).reshape(1, 64, 64, 3)
    arr2 = (np.float32(arr) - 127.5)/127.5
    predict = model6.predict(arr2).reshape(256, 256, 3)
    prediction2 = np.uint8((predict + 1) * 127.5)
    img9 = Image.fromarray(prediction2)
    img9.save('./static/img/10.png')
    buffered = BytesIO()
    img9.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue())
    # img_dict['sr'] = str(img_str)
    str2+='    '
    str2+=str(img_str)

    img10 = im.resize((256, 256), Image.NEAREST)
    img10.save('./static/img/11.png')
    buffered = BytesIO()
    img10.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue())
    # img_dict['near'] = str(img_str)
    str2+='   '
    str2+=str(img_str)

    img11 = im.resize((256, 256), Image.BICUBIC)
    img11.save('./static/img/12.png')
    buffered = BytesIO()
    img11.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue())
    # img_dict['cubic'] = str(img_str_
    str2+='   '
    str2+=str(img_str)
    return str2

@app.route('/resize', methods=['POST'])
def resize_img():
    data = json.loads(request.get_data().decode('utf-8'))
    im = Image.open(requests.get(data['link'], stream=True).raw)
    im = im.resize((800, 800), Image.ANTIALIAS)
    im.save('./static/img/generated/temp.png')
    return port_mobile+':5000/static/img/generated/temp.png'

#@io.on('some event')
#def test_fn(msg):
#    print (msg)
#
#@io.on('my event')
#def handle_my_custom_event(msg):
#    print(msg)
#    
#@io.on('connected')
#def connected():
#    print (request.sid)
#    clients.append(request.sid)
#    
#@io.on('disconnected')
#def disconnect():
#    print ("{} disconnected".format(request.namespace.socket.sessid))
#    clients.remove(request.namespace)
    
if __name__ == '__main__':
    load2()
    app.run(host='0.0.0.0')