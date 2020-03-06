import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Globals
prev_frame = None 
prev_points = None
R_net = None
t_net = None
R_net2 = None
t_net2 = None
keyframe_points = None

parallax = 0.0
parallax_thresh = 30

state = np.zeros((3,))
state_data = [[],[]]

# Algorithm params
feature_params = dict( maxCorners = 250,
                       qualityLevel = 0.06,
                       minDistance = 25,
                       blockSize = 25)
lk_params = dict( winSize  = (100,100),
                  maxLevel = 2,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 500, 1))
clahe = cv.createCLAHE(clipLimit=1.0, 
                       tileGridSize=(8,8))
kernel = (3,3)
points_thresh = 15
dist_thresh = 0.5
image_scale = 3/4

# Camera intrinsic parameter matrix
focal = 2.97 
K = np.zeros((3,3))
K[0,0] = focal
K[1,1] = focal
K[2, 2] = 1
 
# Setup plot for 3D points
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


def preprocess(im):
    # Apply CLAHE then blur 
    im[:,:,0] = cv.GaussianBlur(clahe.apply(im[:,:,0]), kernel, 0) 
    im[:,:,1] = cv.GaussianBlur(clahe.apply(im[:,:,1]), kernel, 0) 
    im[:,:,2] = cv.GaussianBlur(clahe.apply(im[:,:,2]), kernel, 0) 

    # Scale down image
    im = cv.resize(im, (int(im.shape[1]*image_scale), int(im.shape[0]*image_scale)), cv.INTER_CUBIC )

    # Convert image to grayscale
    gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    return gray


def detect_features(im):
    # Detect feactures in image
    points = cv.goodFeaturesToTrack(im, mask=None, **feature_params)
    return points


def visual_odometry(curr_frame):

    # Use globals
    global prev_frame
    global prev_points
    global keyframe_points
    global R_net
    global t_net
    global R_net2
    global t_net2
    global state
    global parallax

    # Preprocess curr frame
    curr_frame = preprocess(curr_frame)

    # First iteration
    if prev_frame is None:
        # Detect first features and create first keyframe
        curr_points = detect_features(curr_frame)
        keyframe_points = curr_points
    else:
        # Track features detected in first image to second image
        curr_points, status,_ = cv.calcOpticalFlowPyrLK(prev_frame, curr_frame, prev_points, None, **lk_params)

        if (np.count_nonzero(status) < points_thresh):
            # Detect new features (or add them on to existing ones??)
            print('Detecting New Features, Ran Out', np.count_nonzero(status))
            curr_points = detect_features(curr_frame)
            keyframe_points = curr_points
        else: 
            # Only keep the points in the images tracked succesfully from the first to the second image
            # Store in seperate variable to preserve shapes of prev, curr, keyframe points i.e  (n, 1, 2) instead of (n,2)
            points_prev = prev_points[status == 1]
            points_curr = curr_points[status == 1]
            points_key = keyframe_points[status == 1] 

            # Update prev, curr, and keyframe points, reshaped
            prev_points = points_prev.reshape((points_prev.shape[0], 1, 2))
            curr_points = points_curr.reshape((points_curr.shape[0], 1, 2))
            keyframe_points = points_key.reshape((points_key.shape[0], 1, 2))   

            # Calculate the mean distance between all corresponding points in each image 
            # This is a rough estimation of how much motion has been captured by the frame sequence
            parallax = parallax + np.mean(np.linalg.norm(points_prev-points_curr, axis=1))         

            #If we have approximately moved enough for an accurate pose estimate
            if (parallax > parallax_thresh):

                # Reset parallax
                parallax = 0.0    

                # Recover change in pose between last keyframe and curr points
                E,_ = cv.findEssentialMat(points_key, points_curr, K)
                _, R, t,_ = cv.recoverPose(E, points_key, points_curr, K)

                if R_net is None and t_net is None:
                    R_net = np.eye(3)
                    t_net = np.zeros((3,1))
                    R_net2 = np.eye(3)
                    t_net2 = np.zeros((3,1))
                
                # Store values of state before updating it (to calculate reprojection of 3D points)
                R_net_prev = R_net.copy()
                t_net_prev = t_net.copy()

                # Projection matrix from origin to last keyframe
                P_key = np.matmul(K, np.hstack((R_net, t_net)))

                # Update state using only pose estimation from essential matrix
                t_net = t_net + np.dot(R_net, t)
                R_net = np.matmul(R, R_net)
                state = t_net.flatten()
                state_data[0].append(state.copy())

                # Projection matrix from last keyframe to curr frame
                P = np.matmul(K, np.hstack((R_net, t_net)))

                # Triangulate from keyframe points to curr points
                points_hom = cv.triangulatePoints(P_key, P, points_key.T.astype(float), points_curr.T.astype(float))
                
                # Convert homogenous coordinates to 3D coordinates
                points_hom[0] = points_hom[0]/points_hom[3]
                points_hom[1] = points_hom[1]/points_hom[3]
                points_hom[2] = points_hom[2]/points_hom[3]
                points_3d = points_hom[:3]

                # Check reprojection error of 3D points onto keyframe
                R_vec,_ = cv.Rodrigues(R_net_prev)
                points_key_reproj,_ = cv.projectPoints(points_3d.T, R_vec, t_net_prev, K, np.zeros((4,)) )
                points_key_reproj = points_key_reproj.reshape((points_key_reproj.shape[0], 2))
                print("Reprojection error of triangulated 3D points onto last keyframe: ", np.linalg.norm(points_key-points_key_reproj))

                # Plot 3D points at every keyframe creation
                # ax.scatter(points_3d[0], points_3d[1], points_3d[2])
                # plt.pause(10)

                # Solve for R, t between keyframe and curr frame using PNP
                _,R_vec, t, inliers = cv.solvePnPRansac(points_3d.T, points_curr, K, np.zeros((4,)))
                
                # Update state data with pose estimate from sovlePnPRansac and triangulated 3D points
                R,_ = cv.Rodrigues(R_vec)
                t_net2 = t_net2 + np.dot(R_net2, t/np.linalg.norm(t))
                R_net2 = np.matmul(R, R_net2)
                state = t_net2.flatten()
                state_data[1].append(state.copy())

                # Make keyframe newly detected points
                curr_points = detect_features(curr_frame)
                keyframe_points = curr_points

    # Draw features on orig image
    for p in curr_points:
        x,y = p.ravel()
        cv.circle(curr_frame, (x,y), 3,  255, -1)

    # Set prev values to curr values
    prev_frame = curr_frame
    prev_points = curr_points

    return curr_frame


##################################
# Video
##################################
video_name = 'dice.mp4'
cap = cv.VideoCapture('./videos/' + video_name)
if (cap.isOpened()== False): 
  print("Error opening video stream or file")
while(cap.isOpened()):
  ret, frame = cap.read()
  if ret == True:
    cv.imshow('Features', visual_odometry(frame))
    if cv.waitKey(25) & 0xFF == ord('q'):
      break
  else: 
    break
cap.release()
cv.destroyAllWindows()


# Plot state data showing (normalized) trajectory
state_data_copy = state_data.copy()

# Plot state data from only essential matrix pose recovery
state_data = np.array(state_data_copy[0]).T
ax.scatter(state_data[0], state_data[1], state_data[2], 'b')
ax.plot(state_data[0], state_data[1], state_data[2],'b')
ax.set_aspect('equal','box')

# Plot state data from triangulation and solvePnpRansac
# state_data = np.array(state_data_copy[1]).T
# ax.scatter(state_data[0], state_data[1], state_data[2], 'r')
# ax.plot(state_data[0], state_data[1], state_data[2], 'r')
# ax.set_aspect('equal','box')


plt.show()




