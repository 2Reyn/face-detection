from random import randrange
import cv2

# # Detector
trained_face_data = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# # For PICTURE

# # Choose img
# img = cv2.imread("CH.jpg")

# # Convert colorful img to gray img
# graycaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# # Detect faces in every scale
# face_coordinates = trained_face_data.detectMultiScale(graycaled_img)

# # if there many faces it will detect them with the help of this cycle
# for (x, y, w, h) in face_coordinates:
# 	cv2.rectangle(img, (x, y), (x + w, y + h), (randrange(128, 256), randrange(128, 256), randrange(128, 256)), 2)

# print(face_coordinates)

# # Show image
# cv2.imshow("Face Detector", img)
# cv2.waitKey()  

# For WEBCAM or Video

webcam  = cv2.VideoCapture(0)

while True:

	# Read currently frame
	succsessful_frame_read, frame = webcam.read()

	# Convert colorful frame to gray frame
	graycaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# Detect faces in every scale
	face_coordinates = trained_face_data.detectMultiScale(graycaled_img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

	# If there many faces it will detect them with the help of this cycle
	for (x, y, w, h) in face_coordinates:
		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 256, 0), 1)

	# Show image
	cv2.imshow("Face Detector", frame)

	# Number inside defind how often frame will change in ms
	key = cv2.waitKey(1)

	if key  == 27:
		break

# Release the VideoCapture objects
webcam.release()

print('Code Completed')