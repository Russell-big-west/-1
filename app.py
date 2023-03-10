import datetime
import logging as rel_log
import os
from pathlib import Path
import shutil
from datetime import timedelta
from flask import *
from time import time
from inference import anime
from predict_photo_fix import photo_fix


from processor.mmt import get_mmt_model
from predict import inpainting

import core.main

UPLOAD_FOLDER = r'./uploads'

ALLOWED_EXTENSIONS = set(['png', 'jpg'])
app = Flask(__name__)
app.secret_key = 'secret!'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

werkzeug_logger = rel_log.getLogger('werkzeug')
werkzeug_logger.setLevel(rel_log.ERROR)

# 解决缓存刷新问题
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = timedelta(seconds=1)


# 添加header解决跨域
@app.after_request
def after_request(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Credentials'] = 'true'
    response.headers['Access-Control-Allow-Methods'] = 'POST'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, X-Requested-With'
    return response


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route('/')
def hello_world():
    return redirect(url_for('static', filename='./index.html'))


@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    file = request.files['file']
    print(datetime.datetime.now(), file.filename)
    if file and allowed_file(file.filename):
        src_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(src_path)
        shutil.copy(src_path, './tmp/ct')
        image_path = os.path.join('./tmp/ct', file.filename)
        print(file.filename)
        return jsonify({'status': 1,
                        'image_url': 'http://127.0.0.1:5003/tmp/ct/' + os.path.basename(image_path),
                        #'draw_url': 'http://127.0.0.1:5003/tmp/draw/' + pid,
                        #'image_info': image_info}
                        })

    return jsonify({'status': 0})


@app.route('/random', methods=['GET', 'POST'])
def random_image():
    print(request.files)
    file = request.files['file']
    
    print("random image: ", file.filename)
    if file and allowed_file(file.filename):
        return jsonify({'status': 1,
                        'image_url': 'http://127.0.0.1:5003/tmp/ct/' + file.filename})

    return jsonify({'status': 0})


@app.route('/inpaint', methods=['GET', 'POST'])
def inpaint_masked_image():
    mask = request.files['mask']
    mask_filename = mask.filename
    img_filename = mask.filename.split('_')[0] + '.jpg'
    print(datetime.datetime.now(), img_filename, mask_filename)
    if mask:
        # save image and mask
        # src_path = os.path.join(app.config['UPLOAD_FOLDER'], img.filename)
        # img.save(src_path)
        # shutil.copy(src_path, './tmp/ct')
        src_path = os.path.join(app.config['UPLOAD_FOLDER'], mask.filename)
        mask.save(src_path)
        shutil.copy(src_path, './tmp/ct')
        
        image_path = os.path.join('./tmp/ct', img_filename)
        mask_path = os.path.join('./tmp/ct', mask_filename)
        ext = image_path.split('.')[-1]
        save_inpainted_fname = str(time()) + '.' + ext
        inpainting(current_app.model, image_path, mask_path, save_inpainted_fname)
        return jsonify({'status': 1,
                        'draw_url': 'http://127.0.0.1:5003/tmp/draw/' + save_inpainted_fname})

    return jsonify({'status': 0})

# 动漫化后台处理函数
@app.route('/anime', methods=['GET', 'POST'])
def anime_image():
    mask = request.files['mask']
    mask_filename = mask.filename
    img_filename = mask.filename.split('_')[0] + '.jpg'
    print(datetime.datetime.now(), img_filename, mask_filename)
    if mask:
        src_path = os.path.join(app.config['UPLOAD_FOLDER'], mask.filename)
        mask.save(src_path)
        shutil.copy(src_path, './tmp/ct')
        
        image_path = os.path.join('./tmp/ct', img_filename)
        mask_path = os.path.join('./tmp/ct', mask_filename)
        anime(image_path, mask_path)
        return jsonify({'status': 1,
                        'draw_url': 'http://127.0.0.1:5003/tmp/ct/' + mask_filename})


# 老照片修复后台处理函数
@app.route('/fix', methods=['GET', 'POST'])
def anime_image():
    mask = request.files['mask']
    mask_filename = mask.filename
    img_filename = mask.filename.split('_')[0] + '.jpg'
    print(datetime.datetime.now(), img_filename, mask_filename)
    if mask:
        src_path = os.path.join(app.config['UPLOAD_FOLDER'], mask.filename)
        mask.save(src_path)
        shutil.copy(src_path, './tmp/ct')
        image_path = os.path.join('./tmp/ct', img_filename)
        mask_path = os.path.join('./tmp/ct', mask_filename)
        photo_fix(image_path, mask_path)
        return jsonify({'status': 1,
                        'draw_url': 'http://127.0.0.1:5003/tmp/ct/' + mask_filename})

@app.route("/download", methods=['GET'])
def download_file():
    # 需要知道2个参数, 第1个参数是本地目录的path, 第2个参数是文件名(带扩展名)
    return send_from_directory('data', 'testfile.zip', as_attachment=True)


# show photo
@app.route('/tmp/<path:file>', methods=['GET'])
def show_photo(file):
    if request.method == 'GET':
        if not file is None:
            image_data = open(f'tmp/{file}', "rb").read()
            response = make_response(image_data)
            response.headers['Content-Type'] = 'image/png'
            return response


if __name__ == '__main__':
    files = [
        'uploads', 'tmp/ct', 'tmp/draw',
        'tmp/image', 'tmp/mask', 'tmp/uploads'
    ]
    for ff in files:
        if not os.path.exists(ff):
            os.makedirs(ff)
    with app.app_context():
        current_app.model = get_mmt_model()
    app.run(host='0.0.0.0', port=5003, debug=True)
