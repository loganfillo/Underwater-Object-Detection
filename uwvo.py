import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


"""
UnderWater Visual Odometry System
"""
class UWVO:


    def __init__(self, feature_quality=0.04, lk_winsize=100, im_scale=1.0):
        """
        Create a UWVO system using given parameters
        """

        # Frame to frame tracking
        self.prev_frame = None 
        self.prev_points = None
        self.keyframe_points = None
        
        # Distance and feature thresholding
        self.parallax = 0.0
        self.parallax_thresh = 30
        self.points_thresh = 15

        # State 
        self.R_net = None
        self.t_net = None
        self.state = np.zeros((3,))
        self.state_data = []

        # Preprocessing 
        self.image_scale = im_scale
        self.clahe = cv.createCLAHE(clipLimit=1.0, 
                       tileGridSize=(8,8))

        # Create intrinsic camera matrix K
        focal = 2.97 
        self.K = np.zeros((3,3))
        self.K[0,0] = focal
        self.K[1,1] = focal
        self.K[2, 2] = 1

        # Algorithm params
        self.feature_params = dict( maxCorners = 250,
                                    qualityLevel = feature_quality,
                                    minDistance = 25,
                                    blockSize = 25)
        self.lk_params = dict( winSize  = (lk_winsize,lk_winsize),
                               maxLevel = 2,
                               criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 500, 1))


    def preprocess(self, im):
        """
        Preprocesses the given image by applying CLAHE, resizing and converting to grayscale
        """

        # Apply CLAHE then blur 
        kernel = (3,3)
        im[:,:,0] = cv.GaussianBlur(self.clahe.apply(im[:,:,0]), kernel, 0) 
        im[:,:,1] = cv.GaussianBlur(self.clahe.apply(im[:,:,1]), kernel, 0) 
        im[:,:,2] = cv.GaussianBlur(self.clahe.apply(im[:,:,2]), kernel, 0) 

        # Scale down image
        newsize = (int(im.shape[1]*self.image_scale), int(im.shape[0]*self.image_scale))
        im = cv.resize(im, newsize, cv.INTER_CUBIC )

        # Convert image to grayscale
        gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
        return gray


    def detect_features(self,im):
        """
        Detect new features in an image (may be modified to add features instead of detecting new ones)
        """
        # Detect feactures in image 
        points = cv.goodFeaturesToTrack(im, mask=None, **self.feature_params)
        return points


    def process(self, curr_frame):
        '''
        Track features from previous frame to current frame, if a parallax exceding a threshold has been
        reached, determine change in pose between previous keyframe and current frame and update the state 
        accordingly. Then triangulate the tracked features in current frame and previous key to achieve 3D
        observations of tracked features. 
        '''

        # Preprocess curr frame
        curr_frame = self.preprocess(curr_frame)

        # First iteration
        if self.prev_frame is None:
            # Detect first features and create first keyframe
            curr_points = self.detect_features(curr_frame)
            self.keyframe_points = curr_points
        else:
            # Track features detected in first image to second image
            curr_points, status,_ = cv.calcOpticalFlowPyrLK(self.prev_frame, curr_frame, self.prev_points, None, **self.lk_params)

            if (np.count_nonzero(status) < self.points_thresh):
                # Detect new features (or add them on to existing ones??)
                print('Detecting New Features, Ran Out', np.count_nonzero(status))
                curr_points = self.detect_features(curr_frame)
                self.keyframe_points = curr_points
            else: 
                # Only keep the points in the images tracked succesfully from the first to the second image
                # Store in seperate variable to preserve shapes of prev, curr, keyframe points i.e  (n, 1, 2) instead of (n,2)
                points_prev = self.prev_points[status == 1]
                points_curr = curr_points[status == 1]
                points_key = self.keyframe_points[status == 1] 

                # Update prev, curr, and keyframe points, reshaped
                self.prev_points = points_prev.reshape((points_prev.shape[0], 1, 2))
                curr_points = points_curr.reshape((points_curr.shape[0], 1, 2))
                self.keyframe_points = points_key.reshape((points_key.shape[0], 1, 2))   

                # Calculate the mean distance between all corresponding points in each image 
                # This is a rough estimation of how much motion has been captured by the frame sequence
                self.parallax = self.parallax + np.mean(np.linalg.norm(points_prev-points_curr, axis=1))         

                #If we have approximately moved enough for an accurate pose estimate
                if (self.parallax > self.parallax_thresh):

                    # Reset self.parallax
                    self.parallax = 0.0    

                    # Recover change in pose between last keyframe and curr points
                    E,_ = cv.findEssentialMat(points_key, points_curr, self. K)
                    _, R, t,_ = cv.recoverPose(E, points_key, points_curr, self.K)

                    if self.R_net is None and self.t_net is None:
                        self.R_net = np.eye(3)
                        self.t_net = np.zeros((3,1))
                    
                    # Store values of self.state before updating it (to calculate reprojection of 3D points)
                    R_net_prev = self.R_net.copy()
                    t_net_prev = self.t_net.copy()

                    # Projection matrix from origin to last keyframe
                    P_key = np.matmul(self.K, np.hstack((self.R_net, self.t_net)))

                    # Update self.state using only pose estimation from essential matrix
                    self.t_net = self.t_net + np.dot(self.R_net, t)
                    self.R_net = np.matmul(R, self.R_net)
                    self.state = self.t_net.flatten()
                    self.state_data.append(self.state.copy())

                    # Projection matrix from last keyframe to curr frame
                    P = np.matmul(self.K, np.hstack((self.R_net, self.t_net)))

                    # Triangulate from keyframe points to curr points
                    points_hom = cv.triangulatePoints(P_key, P, points_key.T.astype(float), points_curr.T.astype(float))
                    
                    # Convert homogenous coordinates to 3D coordinates
                    points_hom[0] = points_hom[0]/points_hom[3]
                    points_hom[1] = points_hom[1]/points_hom[3]
                    points_hom[2] = points_hom[2]/points_hom[3]
                    points_3d = points_hom[:3].T

                    # Check reprojection error of 3D points onto keyframe
                    R_vec_prev,_ = cv.Rodrigues(R_net_prev)
                    points_key_reproj,_ = cv.projectPoints(points_3d, R_vec_prev, t_net_prev, self.K, np.zeros((4,)) )
                    points_key_reproj = points_key_reproj.reshape((points_key_reproj.shape[0], 2))
                    print("Reprojection error of triangulated 3D points onto last keyframe: ", np.linalg.norm(points_key-points_key_reproj))

                    # Plot 3D points at every keyframe creation
                    # ax.scatter(points_3d.T[0], points_3d.T[1], points_3d.T[2])
                    # plt.pause(10)

                    # Solve for R, t between keyframe and curr frame using PNP
                    _,R_vec, t, inliers = cv.solvePnPRansac(points_3d, points_curr, self.K, np.zeros((4,)))
                    
                    # Make keyframe newly detected points
                    curr_points = self.detect_features(curr_frame)
                    self.keyframe_points = curr_points

        # Draw features on orig image
        for p in curr_points:
            x,y = p.ravel()
            cv.circle(curr_frame, (x,y), 3,  255, -1)

        # Set prev values to curr values
        self.prev_frame = curr_frame
        self.prev_points = curr_points

        return curr_frame

def plot_state(state_data):

    # Make sure we have state data
    if len(state_data) == 0:
        return 

    # Setup plot for 3D points
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot state data from essential matrix pose recovery
    state_data = np.array(state_data).T
    ax.scatter(state_data[0], state_data[1], state_data[2], 'b')
    ax.plot(state_data[0], state_data[1], state_data[2],'b')
    ax.set_aspect('equal','box')
    plt.show()

def main():

    # Create visual odometry object
    uwvo = UWVO(feature_quality=0.04, lk_winsize=100, im_scale=3.0/4)

    # Run the visual odometry process on the input video 
    video_name = 'gate.mov'
    cap = cv.VideoCapture('./videos/' + video_name)
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            cv.imshow('Features', uwvo.process(frame))
            if cv.waitKey(25) & 0xFF == ord('q'):
                break
        else: 
            break
    cap.release()
    cv.destroyAllWindows()

    # Plot the state
    plot_state(uwvo.state_data)

# Run main
if __name__ == '__main__':
    main()

