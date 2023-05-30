# The primary goal of this work is to build up a Model of Skin Cancer Detection System utilizing Machine Learning Algorithms. After experimenting with many different architectures for the CNN model It is found that adding the BatchNormalization layer after each Dense, and MaxPooling2D layer can help increase the validation accuracy. In future, a mobile application can be made.
from flask import Flask, request, render_template
from PIL import Image
import numpy as np
import skin_cancer_detection as SCD
import uuid
import os

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def runhome():

    return render_template("index.html")


# The primary goal of this work is to build up a Model of Skin Cancer Detection System utilizing Machine Learning Algorithms. After experimenting with many different architectures for the CNN model It is found that adding the BatchNormalization layer after each Dense, and MaxPooling2D layer can help increase the validation accuracy. In future, a mobile application can be made.


@app.route("/showresult", methods=["GET", "POST"])
def show():
    name = request.form['name']
    age = request.form['age']
    gender = request.form['gender']
    phone = request.form['phone']
    email = request.form['email']

    
    filename = str(uuid.uuid4()) + '.jpg'
    filepath = os.path.join('static', filename)

    img_file = request.files['image']
    img_file.save(filepath)

    inputimg = Image.open(img_file)
    inputimg = inputimg.resize((28, 28))
    img = np.array(inputimg).reshape(-1, 28, 28, 3)
    result = SCD.model.predict(img)

    result = result.tolist()
    print(result)
    max_prob = max(result[0])
    class_ind = result[0].index(max_prob)
    print(class_ind)
    result = SCD.classes[class_ind]

    if class_ind == 0:
        info = "Actinic keratosis also known as solar keratosis or senile keratosis are names given to intraepithelial keratinocyte dysplasia. As such they are a pre-malignant lesion or in situ squamous cell carcinomas and thus a malignant lesion."

    elif class_ind == 1:
        info = "Basal cell carcinoma is a type of skin cancer. Basal cell carcinoma begins in the basal cells — a type of cell within the skin that produces new skin cells as old ones die off.Basal cell carcinoma often appears as a slightly transparent bump on the skin, though it can take other forms. Basal cell carcinoma occurs most often on areas of the skin that are exposed to the sun, such as your head and neck"
    elif class_ind == 2:
        info = "Benign lichenoid keratosis (BLK) usually presents as a solitary lesion that occurs predominantly on the trunk and upper extremities in middle-aged women. The pathogenesis of BLK is unclear; however, it has been suggested that BLK may be associated with the inflammatory stage of regressing solar lentigo (SL)1"
    elif class_ind == 3:
        info = "Dermatofibromas are small, noncancerous (benign) skin growths that can develop anywhere on the body but most often appear on the lower legs, upper arms or upper back. These nodules are common in adults but are rare in children. They can be pink, gray, red or brown in color and may change color over the years. They are firm and often feel like a stone under the skin. "
   
    return render_template("reults.html", result=result, info=info,name=name,img_file=filename,age=age,gender=gender,phone=phone,email=email)


if __name__ == "__main__":
    app.run()
