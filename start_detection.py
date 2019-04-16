import cv2
import sys
import boto3
import time
import os
import requests
import json
import random
from datetime import datetime, timedelta
import threading
from config import *
from PIL import Image

start_time = datetime.now()
cur_time = int(time.time()) * 1000

# cascPath = sys.argv[1] #path for Cascade Classifier for faces ie frontalface.xml
# frontalface.xml absolute path
# frontalfaceXML = os.path.abspath("frontalfacealt.xml")
frontalfaceXML = "frontalface.xml"
print('frontalfaceXML ',frontalfaceXML)
faceCascade = cv2.CascadeClassifier(frontalfaceXML)

# rekognition = boto3.client('rekognition', region_name=Region.US_EAST_1,
#                            aws_access_key_id=Config.AWS_ACCESS_KEY_ID,
#                            aws_secret_access_key=Config.AWS_SECERT_ACCESS_KEY)
# dynamodb = boto3.client('dynamodb', region_name=Region.US_WEST_2,
#                         aws_access_key_id=Config.AWS_ACCESS_KEY_ID,
#                         aws_secret_access_key=Config.AWS_SECERT_ACCESS_KEY)


location = 'XXXXX'


# def crop(img, x, y, w, h, scale_x, scale_y):
#     return img[y - scale_y:y + h + scale_y, x - scale_x:x + w + scale_x]


# def upload(when, frame, faces):
#     pass
#     if not os.path.isdir("checkin_pics"):
#         os.mkdir("checkin_pics")
#     if not os.path.isdir("checkin_pics/" + str(when)):
#         os.mkdir("checkin_pics/" + str(when))

#     url = "https://yymproxqui.execute-api.us-west-2.amazonaws.com/v1/face-recognition"
#     data = dict()
#     i = 0
#     for (x, y, w, h) in faces:
#         # Crop image so just face is in viewable
#         cropped = crop(frame, x, y, w, h, 40, 40)

#         cur_now = str(time.time())
#         # cv2.imwrite("checkin_pics/" + str(when) + "/face-" + cur_now + ".jpg", frame)
#         # img = Image.open("checkin_pics/" + str(when) + "/face-" + cur_now + ".jpg")
#         # cropped = img.crop(((y - 40),( y + h + 40) ,( x - 40) ,(x + w + 40)))

#         if len(cropped) < 0:
#             return None
#         cv2.imwrite("checkin_pics/" + str(when) + "/face-" + cur_now + ".jpg", cropped)
#         # Get bytestream of image
#         b = ''
#         with open("checkin_pics/" + str(when) + "/face-" + cur_now + ".jpg", "rb") as imageFile:
#             f = imageFile.read()
#             if len(f) > 100:
#                 # response = rekognition.search_faces_by_image(
#                 #             CollectionId='alec_collection',
#                 #             Image={'Bytes':f}
#                 #         )

#                 # for match in response['FaceMatches']:
#                 #     # print ('faceID,confidence = ',match['Face']['FaceId'],match['Face']['Confidence'])

#                 #     face = dynamodb.get_item(
#                 #         TableName='fndy_object_info_dev',
#                 #         Key={'deviceId': {'S': match['Face']['FaceId']}}
#                 #         )

#                 #     if 'Item' in face:
#                 #         print ('face detected >>',face['Item']['a.name']['S'])
#                 #         # faceText.append(face['Item']['a.name']['S'])
#                 #         print('Confidence level >>',match['Face']['Confidence'])
#                 #     else:
#                 #         print ('no match found in person lookup')

#                 # POST data generate
#                 data['image'] = list(f)
#                 data['name'] = "face-" + cur_now + ".jpg"
#                 data['logTime'] = str(datetime.now())
#                 # 'IN' for checked in | 'OUT' for checked out
#                 data['checked'] = 'IN'
#                 data['zone'] = 'Lab'  # Zone name
#                 data['logEpoch'] = str(int(time.time()) * 1000)

#                 # attempt to POST face data to AWS api
#                 try:
#                     # print('data ',data)
#                     response = requests.post(url, data=json.dumps(data))
#                 except ConnectionError as e:
#                     print("waiting for internet connection...")

#     print("done... waiting 2 seconds to detect more faces")
#     # time.sleep(3)


cv2.namedWindow("Live_video")
vc = cv2.VideoCapture(0)

if vc.isOpened():  # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False
period = timedelta(seconds=5)
next_time = start_time
seconds = 0
time_t = start_time

while rval:
    cv2.imshow("Live_video", frame)
    rval, frame = vc.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=7,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    # raw_capture.truncate(0)

    if len(faces) > 0:
        # faceText = faces
        # i = len(faceText)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
            # font = cv2.FONT_HERSHEY_SIMPLEX
            # cv2.putText(frame,faceText[i],(x+w-130,y+h+30), font, 0.5, (0,255,0), 1, cv2.LINE_AA)
            # i = i+1
        # print(faces)
        # if next_time < time_t:
        #     next_time = next_time + period + timedelta(seconds=60 * 3)

        # else:
        #     # print('next_time ',next_time)
        #     time_t = time_t + period + timedelta(seconds=60 * 3)
        #     # If faces are detected upload image in new thread
        #     # thr = threading.Thread(target=upload, args=(cur_time, frame, faces), kwargs={})
        #     if thr.is_alive():
        #         thr.join()
        #     else:
        #         thr.start()
            # if faces are detected upload image without thread
            # upload(cur_time, frame, faces)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vc.release()
cv2.destroyWindow("Live_video")
