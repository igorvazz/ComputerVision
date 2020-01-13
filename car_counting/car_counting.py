import numpy as np
import cv2

cap = cv2.VideoCapture('overpass.mp4') 

# take first frame of the video
ret,frame = cap.read()
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

subtractor = cv2.createBackgroundSubtractorKNN(history=10, dist2Threshold=600, detectShadows=False)
kernel = np.ones((5,5),np.uint8)

# left lane mask
vertices1 = np.array([[0,490],[482,300],[644,300],[540,720],[0,720]],np.int32)
pts_mask1 = vertices1.reshape((-1,1,2))

# right lane mask
vertices2 = np.array([[764,720],[673,300],[773,300],[1255,720]],np.int32)
pts_mask2 = vertices2.reshape((-1,1,2))

# frame with zeros to create a mask
black_frame = np.zeros((frame_height,frame_width,3), np.uint8)

# counters of the cars in each lane
cars_left_side = 0
cars_right_side = 0

while True:
    ret, frame = cap.read()

    if ret == True:

        frame_orig = frame.copy()
        
        # creates mask of a region of interest
        cv2.fillPoly(black_frame,[pts_mask1],color=(255,255,255))
        cv2.fillPoly(black_frame,[pts_mask2],color=(255,255,255))
        
        # applies bitwise multiplication from mask with frame
        frame = cv2.bitwise_and(frame, black_frame)

        # gaussian blur
        frame_blur = cv2.GaussianBlur(frame,(5,5),10)
        
        # apply background subtractor
        frame_bg_subtractor = subtractor.apply(frame_blur)
            
        
        # apply a closing kernel followed by binary threshold
        frame_threshold = cv2.morphologyEx(frame_bg_subtractor, cv2.MORPH_CLOSE, kernel)
        
        # apply a opening kernel 
        frame_opening = cv2.morphologyEx(frame_threshold, cv2.MORPH_OPEN, kernel)
        
        # draw a line on left lane
        cv2.line(frame_orig, (70,555), (540,555), (0,60,255), 5)
        
        # draw a line on right lane
        cv2.line(frame_orig, (703,464), (937,464), (255,127,0), 5)
        
        # contour algorithm over the foreground detected with background subtraction
        contours, hierarchy = cv2.findContours(frame_opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        minarea = 1000
        maxarea = 40000
        cxx = np.zeros(len(contours))
        cyy = np.zeros(len(contours)) 
        for i in range(len(contours)):  # cycles through all contours in current frame
            if hierarchy[0, i, 3] == -1:  # using hierarchy to only count parent contours (contours not within others)
                area = cv2.contourArea(contours[i])  # area of contour
                if minarea < area < maxarea:  # area threshold for contour
                    # calculating centroids of contours
                    cnt = contours[i]
                    M = cv2.moments(cnt)
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    # gets bounding points of contour to create rectangle
                    # x,y is top left corner and w,h is width and height
                    x, y, w, h = cv2.boundingRect(cnt)
                    # creates a rectangle around contour
                    cv2.rectangle(frame_orig, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    # Prints centroid text in order to double check later on
                    cv2.putText(frame_orig, str(cx) + "," + str(cy), (cx + 10, cy + 10), cv2.FONT_HERSHEY_SIMPLEX,.3, (0, 0, 255), 1)
                    cv2.drawMarker(frame_orig, (cx, cy), (125, 255, 125), cv2.MARKER_CROSS, markerSize=6, thickness=3,line_type=cv2.LINE_8)
                    
                    # left lane line (70,555), (540,555)
                    if((70 < cx < 540) and ( 550 < cy < 565)):
                        cars_left_side += 1
                        
                    # right lane line (703,464), (927,464)
                    if((703 < cx < 967) and (460 < cy < 466)):
                        cars_right_side += 1
         
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame_orig,text=str(cars_left_side),org=(100,500), fontFace=font,fontScale= 4,color=(0,0,255),thickness=8,lineType=cv2.LINE_AA)
         
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame_orig,text=str(cars_right_side),org=(1000,500), fontFace=font,fontScale= 4,color=(255,0,0),thickness=8,lineType=cv2.LINE_AA)
   
        # shows result
        cv2.namedWindow('frame_orig',cv2.WINDOW_NORMAL)  
        cv2.imshow('frame_orig',frame_orig)
          
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break
        
cap.release()
cv2.destroyAllWindows()


