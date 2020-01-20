import pandas as pd
import cv2

IMAGE_DIR = "/Users/james.neve/Data/Attractiveness/FaceSamples/%s"
FACES_DIR = "/Users/james.neve/Data/Attractiveness/ExtractedFaces/"

CASCADE="Face_cascade.xml"
FACE_CASCADE=cv2.CascadeClassifier(CASCADE)

df = pd.read_csv(IMAGE_DIR % "likes.csv")

for index, row in df.iterrows():
    image_path = IMAGE_DIR % row['image']
    image = cv2.imread(image_path)
    image_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = FACE_CASCADE.detectMultiScale(image_grey, scaleFactor=1.16, minNeighbors=5, minSize=(25, 25), flags=0)

    i = 0
    for x, y, w, h in faces:
        sub_img = image[y - 3:y + h + 3, x - 3:x + w + 3]
        try:
            resized_img = cv2.resize(sub_img, (500, 500), interpolation = cv2.INTER_AREA)
            cv2.imwrite("%s%s_%i.jpg" % (FACES_DIR, row['image_id'], i), resized_img)
            i += 1
        except cv2.error:
            continue
