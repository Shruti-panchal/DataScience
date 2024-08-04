from flask import Flask, render_template, request, redirect, url_for
import os
from PIL import Image
import cv2
import imutils
from skimage.metrics import structural_similarity as ssim

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/images/uploaded/'

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file_upload' not in request.files:
            return redirect(request.url)
        file = request.files['file_upload']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'doc.jpeg')
            file.save(filepath)
            return redirect(url_for('compare_images'))      
    return render_template('index.html')  

@app.route('/compare')
def compare_images():
    original = cv2.imread('./static/images/original/myadh.jpeg')
    tampered = cv2.imread('./static/images/uploaded/doc.jpeg')
    original = cv2.resize(original, (250, 160))
    tampered = cv2.resize(tampered, (250, 160))
    original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    tampered_gray = cv2.cvtColor(tampered, cv2.COLOR_BGR2GRAY)
    
    (score, diff) = ssim(original_gray, tampered_gray, full=True)
    diff = (diff * 255).astype("uint8")
    
    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(original, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.rectangle(tampered, (x, y), (x + w, y + h), (0, 0, 255), 2)
    
    original_image = Image.fromarray(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    tampered_image = Image.fromarray(cv2.cvtColor(tampered, cv2.COLOR_BGR2RGB))
    diff_image = Image.fromarray(diff)
    thresh_image = Image.fromarray(thresh)
    
    original_image.save('./static/images/output/original.png')
    tampered_image.save('./static/images/output/tampered.png')
    diff_image.save('./static/images/output/diff.png')
    thresh_image.save('./static/images/output/thresh.png')

    return render_template('result.html', ssim_score=score)

if __name__ == '__main__':
    app.run(debug=True)    