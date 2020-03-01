import cv2 as cv
import numpy as np

def preprocess(src):
    # Apply CLAHE and Gaussian on each RGB channel then downsize
    clahe = cv.createCLAHE(clipLimit=1.0, tileGridSize=(8,8))
    bgr = cv.split(src)
    kernel = (3,3)
    bgr[0] = cv.GaussianBlur(clahe.apply(bgr[0]), kernel, 0) 
    bgr[1] = cv.GaussianBlur(clahe.apply(bgr[1]), kernel, 0) 
    bgr[2] = cv.GaussianBlur(clahe.apply(bgr[2]), kernel, 0) 
    src = cv.merge(bgr)
    src = cv.resize(src, (int(src.shape[1]/3), int(src.shape[0]/3)), cv.INTER_CUBIC )
    return src

def gradient(src):
    # Compute gradient using grayscale image
    scale = 1
    delta = 0
    ddepth = cv.CV_16S
    grad_x = cv.Sobel(src, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)
    grad_y = cv.Sobel(src, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)
    abs_grad_x = cv.convertScaleAbs(grad_x)
    abs_grad_y = cv.convertScaleAbs(grad_y)
    grad = np.expand_dims(cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0), axis=2)
    return grad

def cluster(src):
    # Compute gradient of grayscale version of source
    hsv = cv.cvtColor(src, cv.COLOR_BGR2HSV)
    grad = gradient(hsv[:,:,1])

    # Create features where each feature vector is [hue, sat, val, grad_magnitude, color_mask]
    err = 30
    upper = (97+err, 56+err, 158+err)
    lower = (97-err, 56-err, 158-err)
    color_mask = cv.inRange(hsv, lower, upper)
    lab = cv.cvtColor(src, cv.COLOR_BGR2LAB)
    features = cv.merge([hsv, grad, color_mask ]).reshape((hsv.shape[0]*hsv.shape[1], 5)) 
    features_float = np.float32(features)

    # K Means segmentation, cluster pixels using k means then segment image using grayscale values
    num_clusters = 4
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 1.0)
    ret,labels,center=cv.kmeans(features_float,num_clusters,None,criteria,3,cv.KMEANS_RANDOM_CENTERS)
    labels = labels.reshape(hsv.shape[0], hsv.shape[1])
    labelled_image = (labels.astype(np.float)/(num_clusters-1))
    labelled_image = (labelled_image*255).astype(np.uint8)
    return grad,  labelled_image

def morphological(src):
    # Dilation then erosion to smooth segmentation
    # TODO: figure out if these are needed at all
    kernel = np.ones((3,3), np.uint8)
    dilated = cv.dilate(src, kernel, iterations=1)
    eroded = cv.erode(dilated, kernel, iterations=1)
    return eroded

def convex_hulls(src):
    # Define convex hull around contours of each cluster
    drawing = np.zeros((src.shape[0], src.shape[1], 3), np.uint8)
    for i in np.unique(src):
        bin = np.where(src!= i, 0, 255).astype(np.uint8)
        if np.sum(bin != 0) >= (bin.shape[0]*bin.shape[1]/2):
            continue 
        # First find contours in the image
        contours, hierarchy = cv.findContours(bin, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        hulls = []
        # Create a convhex hull around each connected contour
        for j in range(len(contours)):
            hulls.append(cv.convexHull(contours[j], False))
        big_hulls = []
        # Get the hulls which are within some range (< 1/4 of image, > 1/5000 of image)
        for hull in hulls:
            con = hull.reshape(hull.shape[0],2).transpose()
            x = con[0] 
            y = con[1] 
            con = np.array([x, y]).transpose().reshape((-1,1,2))
            if (np.max(x)-np.min(x))*(np.max(y)-np.min(y)) > src.shape[0]*src.shape[1]/5000 and (np.max(x)-np.min(x))*(np.max(y)-np.min(y)) < src.shape[0]*src.shape[1]/4:
                blank = np.zeros((src.shape[0], src.shape[1]), np.uint8)
                cv.polylines(blank, [con], True, 255)
                #This displays each hull seperately on a blank image 
                # cv.imshow('Segmented', blank)
                # cv.waitKey(0)
                big_hulls.append(hull)
        # Display all big hulls 
        for j in range(len(big_hulls)):
            cv.drawContours(drawing, big_hulls, j, (0,0,255), 1, 8)
    return drawing

def segmentation(src):
    src = preprocess(src)
    hsv, src = cluster(src)
    return hsv, src

##################################
# Video
##################################
cap = cv.VideoCapture('./videos/gate.mp4')
if (cap.isOpened()== False): 
  print("Error opening video stream or file")
while(cap.isOpened()):
  ret, frame = cap.read()
  if ret == True:
    other, seg = segmentation(frame)
    # seg = morphological(seg)
    con = convex_hulls(seg)
    cv.imshow('Processed', preprocess(frame))
    cv.imshow('other', other)
    cv.imshow('Segmented', seg)
    cv.imshow('Convex Hulls', con)
    if cv.waitKey(25) & 0xFF == ord('q'):
      break
  else: 
    break
cap.release()
cv.destroyAllWindows()

##################################
# Single Image
##################################
# src = cv.imread('./imgs/medium.jpg',1)
# hsv, seg = segmentation(src)
# seg = morphological(seg)
# con = convex_hulls(seg)
# cv.imshow('Source', preprocess(src))
# cv.imshow('Segmented', seg)
# cv.imshow('color_mask', hsv)
# cv.imshow('Convex hulls', con)
# cv.waitKey(0)