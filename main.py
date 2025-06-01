import cv2
from PIL import Image

image_person = 'person.jpg'
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
image = cv2.imread(image_person)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
eyes = eye_cascade.detectMultiScale(gray)
person = Image.open(image_person)
mask = Image.open('mask2.png')
person = person.convert("RGBA")
mask = mask.convert("RGBA")
person = person.convert("RGBA")
for (x,y,w,h) in eyes:
    mask = mask.resize((w, h))
    person.paste(mask, (x, y), mask)
    person.save("eye_masked.png")
    eye_masked = cv2.imread("eye_masked.png")
    cv2.imshow("eye_masked", eye_masked)
    cv2.waitKey()



#
# mask = mask.convert("RGBA")
# for (x,y,w,h) in eyes:
#     pass
# print(eyes)
# cv2.imshow("Person", image)
# cv2.waitKey()
