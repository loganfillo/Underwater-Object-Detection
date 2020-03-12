import cv2 as cv
import numpy as np
import os
import pickle


class GateDetector:
    """
    A class for detecting an underwater gate
    """


    def __init__(self, num_clusters=4, im_resize=1.0/4, debug=False):
        self.num_clusters = num_clusters
        self.im_resize = im_resize
        self.debug = debug
        self.gate_dims = (0,0,0,0)

        with open(os.path.join(os.getcwd(), 'pickle/model.pkl'), 'rb') as file:
            self.model = pickle.load(file)


    def detect(self, src):
        """
        Detects the gate in a raw image

        @param src: Raw underwater image containing the gate

        @returns An image containing the bounding box around a gate, if it can find it 
        """
        pre = self.preprocess(src)
        seg = self.segment(pre)
        morph = self.morphological(seg)
        hulls = self.create_convex_hulls(morph)
        gate = self.bound_gate_using_poles(hulls, src)

        return gate


    def preprocess(self, src):
        """
        Preprocesses the source image to adjust for underwater artifacts


        @param src: A raw unscaled image

        @returns The preprocessed image
        """
        # Apply CLAHE and Gaussian on each RGB channel then downsize
        clahe = cv.createCLAHE(clipLimit=1.0, tileGridSize=(8,8))
        bgr = cv.split(src)
        kernel = (3,3)
        bgr[0] = cv.GaussianBlur(clahe.apply(bgr[0]), kernel, 0) 
        bgr[1] = cv.GaussianBlur(clahe.apply(bgr[1]), kernel, 0) 
        bgr[2] = cv.GaussianBlur(clahe.apply(bgr[2]), kernel, 0) 
        src = cv.merge(bgr)
        src = cv.resize(src, (int(src.shape[1]*self.im_resize), int(src.shape[0]*self.im_resize)), cv.INTER_CUBIC )
        return src


    def gradient(self, src):
        """
        Computes the sobel gradient of a source image

        @param src: A grayscale image

        @returns The sobel gradient of the image
        """
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


    def segment(self, src):
        """
        Constructs features from a source image and clusters the features to produce a segmented image

        @param src: A preprocessed image

        @returns A segmented grayscale image
        """

        # Compute gradient on saturation channel of image (seems to have best response to pole)
        hsv = cv.cvtColor(src, cv.COLOR_BGR2HSV)
        grad = self.gradient(hsv[:,:,1])

        # Create binary image of color mask on hue 
        err = 15
        upper = 15 + err 
        lower = 15 - err 
        color_mask = cv.inRange(hsv[:,:,0], lower, upper)

        # Create features where each feature vector is [hue, sat, val, grad_magnitude, color_mask]
        features = cv.merge([hsv, grad, color_mask]).reshape((hsv.shape[0]*hsv.shape[1], 5)) 
        features_float = np.float32(features)

        # K Means segmentation, cluster pixels using k means then segment image using grayscale values
        num_clusters = self.num_clusters
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 1.0)
        ret,labels,center=cv.kmeans(features_float,num_clusters,None,criteria,3,cv.KMEANS_RANDOM_CENTERS)
        labels = labels.reshape(hsv.shape[0], hsv.shape[1])
        labelled_image = (labels.astype(np.float)/(num_clusters-1))
        labelled_image = (labelled_image*255).astype(np.uint8)

        return  labelled_image


    def morphological(self, src):
        """
        Smooths a segmented image with morphological operations

        @param src: A segmented grayscale image

        @returns A morphologically smoothed image
        """
        # Dilation then erosion to smooth segmentation
        kernel = np.ones((3,3), np.uint8)
        dilated = cv.dilate(src, kernel, iterations=1)
        eroded = cv.erode(dilated, kernel, iterations=1)
        return eroded


    def create_convex_hulls(self, src):
        """
        Creates a set of convex hulls which come from all the binary images of the source image and which are of an 
        appropriate size to be a pole of the gate

        @params src: A segmented grayscale image

        @returns A set of convex hulls where each hull is a set of 2D points
        """
        # Search over the binary images associated to each cluster
        hulls = []
        right_size_hulls = []
        for i in np.unique(src):

            # Create binary image
            bin = np.where(src!= i, 0, 255).astype(np.uint8)

            # If a cluster takes up more than half the screen, it most likely does not contain the poles, skip to next cluster
            if np.sum(bin != 0) >= (bin.shape[0]*bin.shape[1]/2):
                continue

            # First find contours in the image
            _, contours, _ = cv.findContours(bin, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

            # Create a convex hull around each connected contour
            for j in range(len(contours)):
                hulls.append(cv.convexHull(contours[j], False))

            # Get the hulls whose area is within some range dependant on the src image
            for hull in hulls:

                # Get x,y coordinates of hull points
                con = hull.reshape(hull.shape[0],2).transpose()
                x = con[0] 
                y = con[1] 
                con = np.array([x, y]).transpose().reshape((-1,1,2))
                
                # Approximation of hull area
                hull_area = (np.max(x)-np.min(x))*(np.max(y)-np.min(y))

                im_size = src.shape[0]*src.shape[1]
                upper_range = 1.0/7
                lower_range = 1.0/5000
                if (hull_area > im_size*lower_range and hull_area < im_size*upper_range):
                    right_size_hulls.append(hull)

        return right_size_hulls


    def bound_gate_using_poles(self, hulls, src):
        """
        Finds the convex hulls associated to the poles and uses this to draw a bounding box around the poles 
        of the gate

        @param hulls: A set of the convex hulls to search
        @param src: The raw image 

        @returns The raw image with the bounding box around the gate location drawn on
        """

        # Resize src to match the image the hulls were found on
        src = cv.resize(src, (int(src.shape[1]*self.im_resize), int(src.shape[0]*self.im_resize)), cv.INTER_CUBIC )


        # Feature helpers
        def ellispe_features(hull):
            angle = 0
            MA = 1
            ma = 1
            try:
                (x,y),(MA,ma),angle = cv.fitEllipse(hull)
            except:
                pass
            return MA, ma, angle

        def contour_features(hull):
            hull_area = cv.contourArea(hull)
            x,y,w,h = cv.boundingRect(hull)
            rect_area = w*h
            aspect_ratio =  float(w)/h
            return hull_area, rect_area, aspect_ratio

        def featureize(hulls):
            X = []
            
            for hull in hulls:
                features = []

                MA, ma, angle = ellispe_features(hull)
                hull_area, rect_area, aspect_ratio = contour_features(hull)

                axis_ratio = float(MA)/ma
                extent = float(hull_area)/rect_area
                angle = np.abs(np.sin(angle *np.pi/180))

                features.append(axis_ratio)
                features.append(extent)
                features.append(angle)
                features.append(aspect_ratio)

                X.append(features)
            
            return np.asarray(X).astype(float)

        # Featureize hulls
        X_hat = featureize(hulls)

        # Predict 
        y_hat = self.model.predict(X_hat)

        # Get pole hulls
        pole_hulls  = np.array(hulls)[y_hat == 1]

        # Get 2D array of hull points assocaited to a pole
        hull_points = np.empty((0, 2))
        for hull in pole_hulls:
            hull = hull.reshape(hull.shape[0], 2)
            hull_points = np.vstack((hull_points,hull))

        # If we have detected a hull associated to a pole
        if len(hull_points > 0):

            # Get most upper left and lower right points of pole hulls
            # These are the points with smallest and greatest distance from top left (0,0) of image
            min_point = hull_points[np.argmin(np.linalg.norm(hull_points, axis=1))]
            max_point = hull_points[np.argmax(np.linalg.norm(hull_points, axis=1))]

            # Draw min/max point for debug purposes
            if self.debug:
                src = cv.circle(src, (int(min_point[0]), int(min_point[1])), 7, (0,255,0), 3)
                src = cv.circle(src, (int(max_point[0]), int(max_point[1])), 7, (0,255,255), 3)

            x = int(min_point[0])
            y = int(min_point[1])

            w = int(max_point[0]-x) or 0
            h = int(max_point[1]-y) or 1

            # If the bounding rectangle is more wide than high, most likely we have detected both poles
            if float(w)/h >= 1:
                if self.gate_dims != (0,0,0,0):

                    x_p,y_p,w_p,h_p = self.gate_dims
                    prev_area = w_p*h_p

                    # We check to make sure the area hasn't changed too much between frames to account for outliers
                    if not (w*h > 1.25*prev_area or w*h < 0.75*prev_area):
                        self.gate_dims = (x,y,w,h)
                else:
                    self.gate_dims = (x,y,w,h)

        # Draw the gate if we have detected it
        if self.gate_dims != (0,0,0,0):
            x,y,w,h = self.gate_dims
            src = cv.rectangle(src,(x,y),(x+w,y+h),(0,0,255),2)

        # Uncomment to draw all hulls and pole hulls on src for debug purposes
        if self.debug:
            src = cv.polylines(src, hulls,True, (255,255,255),2)
            src = cv.polylines(src, pole_hulls,True, (0,0,255),2)

        return src


    def display_and_label_hulls(self, hulls, src):
        """
        Displays each hull and allows someone to label the hull as being associated to a pole or not and returns a
        data structure mapping hulls to labels

        @param hulls: The convex hulls 
        @param src: The source image from which teh convex hulls have been created

        @returns A list where each entry is a tuple mapping a hull to it's label
        """
        
        labels = []

        for hull in hulls:

            angle = 0
            MA = 1
            ma = 1
            try:
                (x,y),(MA,ma),angle = cv.fitEllipse(hull)
            except:
                pass
            cosAngle = np.abs(np.cos(angle*np.pi/180))

            # Only human-classify hulls if it is reasonably a vertically oriented rectangle
            # This is a hueristic to not have to waste time clasifying hulls clearly not poles
            if  (cosAngle < 1.75) and (cosAngle > 0.85) and (MA/ma < 0.28):
                cpy = src.copy()
                hull_img = cv.polylines(cpy, [hull], True, (0,0,255), 3)
                cv.imshow("Hull", hull_img)
                keycode = cv.waitKey(0)
                if keycode == 49:
                    labels.append((hull, 0))
                    print("Not a Pole")
                elif keycode == 50:
                    labels.append((hull, 1))
                    print("A Pole!")
                else:
                    raise Exception("Unexpected Key Pressed")
            else:
                labels.append((hull, 0))
        cv.destroyAllWindows()
        return labels


    def create_labelled_dataset(self):
        """
        From the images folder, tries to open the images and for each image, allows the user to label the hulls as being 
        associated to a pole or not, and returns a dictionary of hulls to labels
        
        @returns A dictionary mapping the hulls of each image to their labels
        """

        print("-------------------------------------------------------------------")
        print("                 How to Use the Hull Label Tool")
        print("-------------------------------------------------------------------")
        print("- If a hull is NOT associated to a pole: press the 1 button")
        print("- If a hull IS associated to a pole: press the 2 button")
        print("\n- If any other key is pressed, the program EXITS")
        print("-------------------------------------------------------------------")

        imgs = []
        labels = []
        directory = os.getcwd()
        
        # Get absolute path of all images in the images folder
        for dirpath,_,filenames in os.walk(os.path.join(directory, 'images')):
            for f in filenames:
                imgs.append(os.path.abspath(os.path.join(dirpath, f)))

        # Get the hulls from the segmented image and run the display and label program for each image
        for img in imgs:
            src = cv.imread(img, 1)
            pre = self.preprocess(src)
            seg = self.segment(pre)
            mor = self.morphological(seg)
            hulls = self.create_convex_hulls(seg)
            labels += self.display_and_label_hulls(hulls, pre)
            
        return labels

        
def video(video_name, im_resize=1.0, record=False):
    """
    Test PoleDetector on a video, records the output
    """
    detector = GateDetector(im_resize=im_resize)
    cap = cv.VideoCapture('./videos/' + video_name )
    frame_width = int(cap.get(3)*im_resize)
    frame_height = int(cap.get(4)*im_resize)
    if record:
        out = cv.VideoWriter('videos/output.avi',cv.VideoWriter_fourcc(*'XVID') , 20.0, (frame_width,frame_height))
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            poles = detector.detect(frame)
            if record :
                out.write(poles)
            cv.imshow("Poles",poles)
            if cv.waitKey(25) & 0xFF == ord('q'):
                break
        else: 
            break
    cap.release()
    if record:
        out.release()
    cv.destroyAllWindows()


def image(image_name, im_resize=1.0):
    """
    Test PoleDetector on an image, shows the images at various stages
    """
    src = cv.imread('./images/' + image_name,1)

    detector = GateDetector(im_resize=im_resize, debug=True)

    pre = detector.preprocess(src)
    seg = detector.segment(pre)
    seg = detector.morphological(seg)
    hulls = detector.create_convex_hulls(seg)
    gate = detector.bound_gate_using_poles(hulls, src)
    cv.imshow('Processed', pre)
    cv.imshow('Segmented', seg)
    cv.imshow('Gate', gate)
    cv.waitKey(0)


def label_poles():
    """
    Run Pole label program
    """
    detector = GateDetector(im_resize=3.0/4)
    labels = detector.create_labelled_dataset()

    # Dump labels data to pickle
    with open(os.path.join(os.getcwd(), 'pickle/pole_data2.pkl'), 'wb') as file:
        pickle.dump(labels, file)


if __name__ == '__main__':
    im_resize = 1.0
    video('gate.mp4',im_resize=im_resize, record=True)
    # for i in range(16):
    #     im_name = 'raw_Moment' + str(i) + '.jpg'
    #     image(im_name, im_resize=im_resize)
    # label_poles()
