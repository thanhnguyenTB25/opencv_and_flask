from flask import Flask, request, render_template,flash,Response,redirect, url_for
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
import pytesseract
from camera import Video
UPLOAD_FOLDER = 'static/uploads/'
ALLOWED_EXTENSIONS = {'txt', 'webp', 'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'super secret key'
def processImg(filename,operation):
    print(f"the operation is {operation} and file name is{filename}")
    img = cv2.imread(f"static/uploads/{filename}")
    match operation:
        case "shapeDetect":
            # chuyen anh sang mau xam
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            #phan nguong
            ret, thresh = cv2.threshold(gray, 235, 255, cv2.THRESH_BINARY_INV)

            # tim duong vien
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # ham tinh hinh tron(dien tich + chu vi hinh tron)
            def circularity(contour):
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)
                if perimeter == 0:
                    return 0
                else:
                    return 4 * np.pi * area / (perimeter ** 2)

            # tinh ty le khung hinh(w/h)
            def aspect_ratio(contour):
                x,y,w,h = cv2.boundingRect(contour)
                return float(w)/h

            # tinh do toan ven cua hinh
            def solidity(contour):
                area = cv2.contourArea(contour)
                hull_area = cv2.contourArea(cv2.convexHull(contour))
                if hull_area == 0:
                    return 0
                else:
                    return float(area)/hull_area

            # xac dinh hinh
            def detect_shape(contour):
                shape = ""
                approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)#dem so dinh
                circularity_val = circularity(contour)
                solidity_val = solidity(contour)
                aspect_ratio_val = aspect_ratio(contour)

                if circularity_val > 0.8:
                    shape = "Circle"
                elif len(approx) == 3:
                    shape = "Triangle"
                elif len(approx) == 4:
                    if aspect_ratio_val >= 0.95 and aspect_ratio_val <= 1.05:
                        shape = "Square" 
                    else:
                        shape = "Rectangle"
                elif circularity_val > 0.7 :
                    shape = "Heart"
                elif len(approx) == 10:
                    shape = "Star"
                else:
                    shape = "Unknown"
                return shape

            # vong lap tat cac cac duong vien
            for contour in contours:
                shape = detect_shape(contour)
                cv2.drawContours(img, [contour], 0, (0, 255, 0), 2) # ve duon vien
                x,y = contour[0][0] #toa do chu
                cv2.putText(img, shape, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2) #viet chu
            imgprocess = img.copy()
            cv2.imwrite(f"static/uploads/{filename}",imgprocess)
            return filename
        #===================textDetect=========================
        case "textDetect":
            pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

            # chuyen sang anh xam
            gray1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # phan nguong
            thresh1 = cv2.threshold(gray1, 235, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

            # tim duong vien
            contours1, hierarchy = cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # loc duong vien(nếu kích thước quá nhỏ)
            def filter_contour(c):
                (x, y, w, h) = cv2.boundingRect(c)
                if w < 5 or h < 5:
                    return False
                aspect_ratio = w / float(h)
                if aspect_ratio > 2.5:
                    return False
                area = cv2.contourArea(c)
                if area < 10:
                    return False
                return True

            # loc, giu lai cac duong vien co kha nang la ki tu
            filtered_contours = [c for c in contours1 if filter_contour(c)]

            # Sắp xếp các đường viền từ trái sang phải
            bounding_boxes = [cv2.boundingRect(c) for c in filtered_contours]
            (contours1, bounding_boxes) = zip(*sorted(zip(filtered_contours, bounding_boxes), key=lambda b: b[1][0], reverse=False))

            # xuat ki ty bang cach cat phan gioi han boi duong vien
            characters = []
            for contour in contours1:
                (x, y, w, h) = cv2.boundingRect(contour)
                char_image = thresh1[y:y+h, x:x+w]
                characters.append(char_image)

            # chuyen sang dang text
            text = ''
            for char_image in characters:
                char = pytesseract.image_to_string(char_image, lang='eng')
                text += char
            print(text)
            with open(f"static/uploads/text.txt", 'w') as f:
                f.write(text)
            os.remove(f"static/uploads/{filename}")
            return filename
        # ====================phát hiện mặt======================
        case "faceDetect":
            faceCascade= cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
            imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)   
            faces = faceCascade.detectMultiScale(imgGray,1.1,4)
            for (x,y,w,h) in faces:
                cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            imgprocess = img.copy()
            cv2.imwrite(f"static/uploads/{filename}",imgprocess)  
            return filename     
    pass
# =======================================================================================
@app.route("/")
def home():
    return render_template('index.html')

@app.route("/<string:page_name>")
def page_name(page_name):
    return render_template(page_name)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/edit", methods=['GET','POST'])
def upload_file():
    if request.method == 'POST':
        operation = request.form.get("operation")
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return "error"
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return "no such file"
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            print(filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            processImg(filename,operation)
            flash(f"your image has been processed <a href='static/uploads/{filename}'>here</a>")
            return render_template("index.html")
        return render_template("index.html")

def gen(camera):
    while True:
        frame=camera.get_frame()
        yield(b'--frame\r\n'
       b'Content-Type:  image/jpeg\r\n\r\n' + frame +
         b'\r\n\r\n')

@app.route('/video')
def video():
    return Response(gen(Video()),
    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/display_image', methods=['GET','POST'])
def display_image(filename):
    if request.method == 'POST':
    #print('display_image filename: ' + filename)
        return redirect(url_for('static', filename='uploads/' + filename), code=301)
 

if __name__ == '__main__':
    app.run(debug=True,port=5001)