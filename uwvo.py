import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os


class UWVO:
    """
    UnderWater Visual Odometry System
    """


    def __init__(self, feature_quality=0.04, tracking_winsize=21, im_scale=1.0, min_features=100, cam_focal=1.0):
        """
        Create a UWVO system using given parameters

        @param feature_quality: The feature quality used on harris corner detection
        @param tracking_winsize: The window size to search over when tracking features
        @param im_scale: The resized scale of the image which the algorithm will be run on
        @param min_features: The min number of features that the algorithm tracks
        @param cam_focal: The focal length of the camera
        """

        # Frame to frame tracking
        self.prev_frame = None 
        self.prev_points = None
        self.keyframe_points = None
        
        # Distance and feature thresholding
        self.parallax = 0.0
        self.parallax_thresh = 30
        self.points_thresh = min_features

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
        focal = cam_focal
        self.K = np.zeros((3,3))
        self.K[0,0] = focal
        self.K[1,1] = focal
        self.K[2, 2] = 1

        # Algorithm params
        self.feature_params = dict( maxCorners = 500,
                                    qualityLevel = feature_quality,
                                    minDistance = 7,
                                    blockSize = 7)
        self.lk_params = dict( winSize  = (tracking_winsize,tracking_winsize),
                               maxLevel = 3,
                               criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 30, 0.001))


    def preprocess(self, im):
        """
        Preprocesses the given image by applying CLAHE, resizing and converting to grayscale

        @param im: The raw image
        
        @returns A preprocessed grayscale image
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

        @param src: The grayscale image to detect features on

        @returns The list of 2D points detected
        """
        # Detect feactures in image 
        points = cv.goodFeaturesToTrack(im, mask=None, **self.feature_params)
        return points


    def process(self, curr_frame, scale):
        '''
        Track features from previous frame to current frame, 

        @param curr_frame: The frame that is being processed in the VO algorithm
        @param scale: The magnitude of the translation vector between the previou frame and current frame

        @returns The current frame with features tracked drawn on it
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
                # Detect new features if we fall below the points threshold (or add them on to existing ones??)
                print('Detecting New Features, Ran Out: ', np.count_nonzero(status))
                curr_points = self.detect_features(curr_frame)
                self.keyframe_points = curr_points
            else: 
                # Only keep the points in the images tracked succesfully from the first to the second image
                # Store in seperate variable to preserve shapes of prev, curr, keyframe points i.e  (n, 1, 2) instead of (n,2)
                points_prev = self.prev_points[status == 1]
                points_key = self.keyframe_points[status == 1] 
                points_curr = curr_points[status == 1]

                # Update prev, curr, and keyframe points, reshaped
                self.prev_points = points_prev.reshape((points_prev.shape[0], 1, 2))
                curr_points = points_curr.reshape((points_curr.shape[0], 1, 2))
                self.keyframe_points = points_key.reshape((points_key.shape[0], 1, 2)) 

                # Recover change in pose between last keyframe and curr points
                # E,_ = cv.findEssentialMat(points_key, points_curr, self. K)
                # _, R, t,_ = cv.recoverPose(E, points_key, points_curr, self.K)
                E,_ = cv.findEssentialMat(points_prev, points_curr, self. K)
                _, R, t,_ = cv.recoverPose(E, points_prev, points_curr, self.K)


                if self.R_net is None and self.t_net is None:
                    self.R_net = np.eye(3)
                    self.t_net = np.zeros((3,1))
                
                # Store values of self.state before updating it (to calculate reprojection of 3D points)
                R_net_prev = self.R_net.copy()
                t_net_prev = self.t_net.copy()  

                # Projection matrix from origin to last keyframe
                P_key = np.matmul(self.K, np.hstack((self.R_net, self.t_net)))

                # Update self.state using only pose estimation from essential matrix
                self.t_net = self.t_net + scale*np.dot(self.R_net, t)
                self.R_net = np.matmul(R, self.R_net)
                self.state = self.t_net.flatten()
                self.state_data.append(self.state.copy())

                # Projection matrix from last keyframe to curr frame
                P = np.matmul(self.K, np.hstack((self.R_net, self.t_net)))


                # Calculate the mean distance between all corresponding points in each image 
                # This is a rough estimation of how much motion has been captured by the frame sequence
                self.parallax = self.parallax + np.mean(np.linalg.norm(points_prev-points_curr, axis=1))         

                #If we have approximately moved enough for an accurate pose estimate
                if (self.parallax > self.parallax_thresh):

                    # Reset self.parallax
                    self.parallax = 0.0    

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

                    #Plot 3D points at every keyframe creation
                    # ax.scatter(points_3d.T[0], points_3d.T[1], points_3d.T[2])
                    # plt.pause(0.10)

                    # Solve for R, t between keyframe and curr frame using PNP
                    # _,R_vec, t, inliers = cv.solvePnPRansac(points_3d, points_curr, self.K, np.zeros((4,)))
                    
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


def plot_state(state_data, ground_truth=None):

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
    if ground_truth is not None:
        ax.plot(ground_truth [0], ground_truth [1], ground_truth [2],'r')
    ax.set_aspect('equal','box')
    plt.show()


def video_test():

    # Create visual odometry object
    uwvo = UWVO(feature_quality=0.1, tracking_winsize=4, im_scale=3.0/4, min_features=10, cam_focal=2.97)

    # Run the visual odometry process on the input video 
    video_name = 'gate.mp4'
    cap = cv.VideoCapture('./videos/' + video_name)
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            cv.imshow('Features', uwvo.process(frame, 1.0))
            if cv.waitKey(25) & 0xFF == ord('q'):
                break
        else: 
            break
    cap.release()
    cv.destroyAllWindows()

    # Plot the state
    plot_state(uwvo.state_data)


def kitti_dataset_test():

    # Load pose data
    file_name = os.path.expanduser('~')+'/Subbots/KITTI-Dataset/poses/00.txt'
    lines = []
    with open(file_name) as file:
        lines = file.readlines()
    poses = []
    for line in lines:
        poses.append(line.strip())

    # Helper for recovering scale
    def get_scale(frame_num):
        x,y,z,x_prev,y_prev,z_prev = 0,0,0,0,0,0
        if (frame_num-1 >= 0):
            prev_frame_pose = poses[frame_num-1].split(" ")
            # Parse pose
            x_prev = float(prev_frame_pose[3])
            y_prev = float(prev_frame_pose[7])
            z_prev = float(prev_frame_pose[11])
        frame_pose = poses[frame_num].split(" ")
        # Parse pose
        x = float(frame_pose[3])
        y = float(frame_pose[7])
        z = float(frame_pose[11])
        return np.sqrt((x-x_prev)**2 + (y-y_prev)**2 + (z-z_prev)**2), [x,y,z]


    # Create visual odometry object
    uwvo = UWVO(feature_quality=0.008, tracking_winsize=50, im_scale=1.0, min_features=250, cam_focal=718.8560)

    # Read frames
    folder = os.path.expanduser('~')+ '/Subbots/KITTI-Dataset/imgs/dataset/sequences/00/image_1/'
    num_frames = 4540;
    ground_truth_data = []
    for i in range (num_frames + 1):
        frame_name = folder + str(i).zfill(6) + '.png' 
        frame = cv.imread(frame_name)   
        scale, state = get_scale(i)
        print(state)
        ground_truth_data.append(state)
        cv.imshow('Frame',uwvo.process(frame, scale))
        if cv.waitKey(25) & 0xFF == ord('q'):
            break
    cv.destroyAllWindows()

    ground_truth_data = np.array(ground_truth_data).T

    # Plot the state
    plot_state(uwvo.state_data, ground_truth=ground_truth_data)


# Run main
if __name__ == '__main__':
    video_test()
    # kitti_dataset_test()

