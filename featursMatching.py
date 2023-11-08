import cv2
import numpy as np
import os

path = "imagesQuary"
images = []
classNames = []
orb = cv2.ORB_create(nfeatures=1000)
bf = cv2.BFMatcher()

myList = os.listdir(path)
for cl in myList:

    img = cv2.imread(f'{path}/{cl}', 0)
    images.append(img)
    classNames.append(os.path.splitext(cl)[0])


def findDes(images):
    deslist = []
    for img in images:
        kp1, des1 = orb.detectAndCompute(img, None)

        deslist.append(des1)

    return deslist


def findId(img, desList, thrsh=15):
    kp1, des1 = orb.detectAndCompute(img, None)
    matchList = []
    finalVal = -1
    try:

        for des in desList:

            matches = bf.knnMatch(des, des1, k=2)

            good = []
            for m, n in matches:
                if m.distance < 0.75*n.distance:
                    good.append([m])
            matchList.append(len(good))
        # print(matchList)
    except:
        pass

    if (len(matchList) != 0):
        # print(matchList)
        if max(matchList) > thrsh:
            finalVal = matchList.index(max(matchList))
    # print(finalVal)
    return finalVal


desList = findDes(images)


cap = cv2.VideoCapture(0)
while True:
    success, img = cap.read()

    id = findId(img, desList)
    if (id != -1):

        cv2.putText(img, classNames[id], (50, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    cv2.imshow("img", img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        cv2.destroyAllWindows()
        break
