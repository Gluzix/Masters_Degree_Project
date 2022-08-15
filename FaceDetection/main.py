import cv2

imagePath = "image_136.png"
cascPath = "haarcascade_frontalface_default.xml"

faceCascade = cv2.CascadeClassifier(cascPath)

image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30),
    flags=cv2.CASCADE_SCALE_IMAGE
)

print("Found {0} faces!".format(len(faces)))

for (x, y, w, h) in faces:
    cv2.rectangle(image, (x-70, y-70), (x+w+70, y+h+70), (0, 255, 0), 2)
    # cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cropped_image = image[y-70:y+h+70, x-70:x+w+70]
    cv2.imshow("Faces found", cropped_image)
    cv2.waitKey(0)

cv2.imshow("Faces found", image)
cv2.imwrite("Face_found.png", image)
cv2.waitKey(0)
