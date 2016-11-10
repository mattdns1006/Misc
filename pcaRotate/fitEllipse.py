import cv2
import numpy as np

def fitEllipse(mask, minThresh, maxThresh):

    #Greyscale for ellipse to work
    imgray = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)

    fit_ellipse = False
    while fit_ellipse == False:
        try:
            length_cnt = 0 
            while length_cnt < 6:

                ret,thresh = cv2.threshold(imgray,minThresh,maxThresh,3)
                contours,hierarchy = cv2.findContours(thresh, 1, 2)

                #Find the shape with the most contours i.e the largest (this is the shape/ellipse we want to keep).
                n_contours = [i.shape[0] for i in contours]
                contours_largest = n_contours.index(max(n_contours))

                #Select contours with largest index
                cnt = contours[contours_largest]
                
                #Need at least 5 contour points to fit ellipse
                length_cnt = len(cnt)
                if length_cnt <6:
                    min_thresh -=2

            M = cv2.moments(cnt)
            ellipse = cv2.fitEllipse(cnt)

            fit_ellipse = True
        except ValueError:
            minThresh -= 2

    mask1 = np.zeros_like(imgray)
    cv2.ellipse(mask1,ellipse,(255,255,255),-1)
    
    #cv2.ellipse(mask,ellipse,(255,255,255),10)
    #cv2.imwrite("test.jpg",mask)
    #ipdb.set_trace()

    #Apply mask to all 3 channels of origial mask
    for dim in range(mask.shape[2]):
        mask[:,:,dim] = np.bitwise_and(mask[:,:,dim],mask1)

    return mask


if __name__ == "__main__":
    import ipdb, sys, glob
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    def show(img):
        plt.imshow(img,cmap=cm.gray); plt.show()

    egPaths = glob.glob("/home/msmith/kaggle/whale/imgs/test/m1_5851.jpg*")
    print("Examples of ellipse fitting")
    for path in egPaths:
        mask = cv2.imread(path)
        w, h = cv2.imread(path.replace("m1","w1")).shape[:2]
        mask = cv2.resize(mask,(h,w), interpolation= cv2.INTER_CUBIC)
        maskBefore = mask.copy()
        maskAfter = fitEllipse(mask,25,200)
        show(np.hstack((maskBefore,maskAfter)))
        ipdb.set_trace()

