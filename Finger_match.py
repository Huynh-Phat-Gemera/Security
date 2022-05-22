# Import computer vision library (cv2)
import re
import cv2
import os
import argparse
import time

try:
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required = True, help = "Path to the image")
    args = vars(ap.parse_args())

    # Tạo biến chứa ảnh nguồn đầu vào
    source_image = cv2.imread(args["image"])
    score = 0
    file_name = None
    image = None
    # kp1 = source image, kp2 = target image, mp = match points 
    kp1, kp2, mp = None, None, None
    keypoints = 0

    start_time = time.time()

    # Vòng lặp lấy tất cả ảnh trong thư mục database
    for file in [file for file in os.listdir("database")][:]:
        # Đọc từng ảnh đã lấy
        target_image = cv2.imread("./database/" + file)

        # SIFT (Scale-invariant feature transform) - phương pháp trích chọn đặc trưng
        sift = cv2.SIFT.create()
        # get the key points and description from the source image
        kp1, des1 = sift.detectAndCompute(source_image, None)
        # get the key points and description form the target image
        kp2, des2 = sift.detectAndCompute(target_image, None)
        # get matches for both source and target image descriptors
        matches = cv2.FlannBasedMatcher(dict(algorithm = 1, trees = 10), dict()).knnMatch(des1, des2, k = 2)

        mp = []
        # create a loop for 2 variables p and q
        for p, q in matches:
            if p.distance < 0.1 * q.distance:
                # append the p variable in matches to mp list
                mp.append(p)
                
                if len(kp1) <= len(kp2):
                    keypoints = len(kp1)
                else:
                    keypoints = len(kp2)

                if len(mp) / keypoints * 100 > score:
                    score = len(mp) / keypoints * 100
                    print('The best match: ' + file)
                    # print('The score: ' + str(score))
                    result = cv2.drawMatches(source_image, kp1, target_image, kp2, mp, None)
                    result = cv2.resize(result, None, fx = 2.5, fy = 2.5)
                    cv2.imshow("Result", result)

                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    print ("Elapsed time: {0}".format(elapsed_time) + " [sec]")

                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                    break; 

    if keypoints == 0:
        print('Not match!')
except:
    print ("Error!")