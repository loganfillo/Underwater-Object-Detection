import cv2 as cv
import numpy as np


# Free parameters of the system 
IMG_RESIZE_SCALE = 1.0/3
NUM_CLUSTERS = 5


def preprocess(src):
    # Apply CLAHE and Gaussian on each RGB channel then downsize
    clahe = cv.createCLAHE(clipLimit=1.0, tileGridSize=(8,8))
    bgr = cv.split(src)
    kernel = (3,3)
    bgr[0] = cv.GaussianBlur(clahe.apply(bgr[0]), kernel, 0) 
    bgr[1] = cv.GaussianBlur(clahe.apply(bgr[1]), kernel, 0) 
    bgr[2] = cv.GaussianBlur(clahe.apply(bgr[2]), kernel, 0) 
    src = cv.merge(bgr)
    src = cv.resize(src, (int(src.shape[1]*IMG_RESIZE_SCALE), int(src.shape[0]*IMG_RESIZE_SCALE)), cv.INTER_CUBIC )
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

    # Compute gradient on saturation channel of image (seems to have best response to pole)
    hsv = cv.cvtColor(src, cv.COLOR_BGR2HSV)
    grad = gradient(hsv[:,:,1])

    # Create color mask on HSV image
    err = 30
    upper = (97+err, 56+err, 158+err)
    lower = (97-err, 56-err, 158-err)
    color_mask = cv.inRange(hsv, lower, upper)

    # Create features where each feature vector is [hue, sat, val, grad_magnitude, color_mask]
    features = cv.merge([hsv, grad, color_mask ]).reshape((hsv.shape[0]*hsv.shape[1], 5)) 
    features_float = np.float32(features)

    # K Means segmentation, cluster pixels using k means then segment image using grayscale values
    num_clusters = NUM_CLUSTERS
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 1.0)
    ret,labels,center=cv.kmeans(features_float,num_clusters,None,criteria,3,cv.KMEANS_RANDOM_CENTERS)
    labels = labels.reshape(hsv.shape[0], hsv.shape[1])
    labelled_image = (labels.astype(np.float)/(num_clusters-1))
    labelled_image = (labelled_image*255).astype(np.uint8)

    return color_mask,  labelled_image


def morphological(src):
    # Dilation then erosion to smooth segmentation
    kernel = np.ones((3,3), np.uint8)
    dilated = cv.dilate(src, kernel, iterations=1)
    eroded = cv.erode(dilated, kernel, iterations=1)
    return eroded


def convex_hulls(src, orig):

    # Define convex hull around contours of each cluster
    drawing = np.zeros((src.shape[0], src.shape[1], 3), np.uint8)

    # Scale raw image to be the same as the segmented image on which the convex hulls are computed
    orig = cv.resize(orig, (int(orig.shape[1]*IMG_RESIZE_SCALE), int(orig.shape[0]*IMG_RESIZE_SCALE)), cv.INTER_CUBIC)

    # Search over the binary images associated to each cluster
    for i in np.unique(src):

        # Create binary image
        bin = np.where(src!= i, 0, 255).astype(np.uint8)

        # If a cluster takes up more than half the screen, it most likely does not contain the poles, skip to next cluster
        if np.sum(bin != 0) >= (bin.shape[0]*bin.shape[1]/2):
            continue

        # First find contours in the image
        img, contours, hierarchy = cv.findContours(bin, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        hulls = []

        # Create a convex hull around each connected contour
        for j in range(len(contours)):
            hulls.append(cv.convexHull(contours[j], False))
        pole_hulls = []

        for hull in hulls:
            con = hull.reshape(hull.shape[0],2).transpose()
            x = con[0] 
            y = con[1] 
            con = np.array([x, y]).transpose().reshape((-1,1,2))
            
            # Get the hulls whose area is within some range (< 1/5 of image, > 1/5000 of image for example)
            im_size = src.shape[0]*src.shape[1]
            if (np.max(x)-np.min(x))*(np.max(y)-np.min(y)) > im_size/5000 and (np.max(x)-np.min(x))*(np.max(y)-np.min(y)) < im_size/5:
                
                #This displays each good size hull and coprresponding fitted ellipse seperately on a blank image 
                # blank = np.zeros((src.shape[0]*2, src.shape[1]*2,3), np.uint8)
                # cv.polylines(blank, [con], True, (0,0,255))
                # # ellipse = cv.fitEllipse(con)
                # # cv.ellipse(blank, ellipse,(0,255,0), 1)
                # print(con)
                # cv.imshow('Convex Hull', blank)
                # cv.waitKey(0)

                #Sometime fitEllipse doesn't work on a convex hull, we need dummy values
                angle = 0
                MA = 1
                ma = 1
                try:
                    (x,y),(MA,ma),angle = cv.fitEllipse(con)
                except:
                    pass
                cosAngle = np.abs(np.cos(angle*np.pi/180))
                print(MA/ma, cosAngle)

                # Only add hull to pole hulls if it is reasonably a vertically oriented rectangle (this will be a ML model)
                if  (cosAngle < 1.2) and (cosAngle > 0.98) and (MA/ma < 0.25):
                    pole_hulls.append(hull)

        # Display all pole hulls on original image
        for j in range(len(pole_hulls)):
            cv.drawContours(orig,pole_hulls, j, (0,0,255), 1, 8)
            
    return orig

def segmentation(src):
    src = preprocess(src)
    other, src = cluster(src)
    return other, src

##################################
# Video
##################################
video_name = 'gate.mov'
cap = cv.VideoCapture('./videos/' + video_name )
if (cap.isOpened()== False): 
  print("Error opening video stream or file")
while(cap.isOpened()):
  ret, frame = cap.read()
  if ret == True:
    other, seg = segmentation(frame)
    # seg = morphological(seg)
    con = convex_hulls(seg, frame)
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
# other, seg = segmentation(src)
# # seg = morphological(seg)
# con = convex_hulls(seg, src)
# cv.imshow('Source', preprocess(src))
# cv.imshow('Segmented', seg)
# cv.imshow('color_mask', other)
# cv.imshow('Convex hulls', con)
# cv.waitKey(0)