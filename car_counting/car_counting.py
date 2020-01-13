import numpy as np
import cv2

cap = cv2.VideoCapture('overpass.mp4') 
frames_count, fps, width, height = cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FPS), cap.get(
    cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

print(frames_count, fps, width, height)

# take first frame of the video
ret,frame = cap.read()

# height , width , layers =  frame.shape
# new_h=int(height/2)
# new_w=int(width/2)
# frame = cv2.resize(frame, (new_w, new_h)) 

#subtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=200, detectShadows=True)
subtractor = cv2.createBackgroundSubtractorKNN(history=10, dist2Threshold=600, detectShadows=False)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

kernel = np.ones((5,5),np.uint8)
# #save video
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (frame_width,frame_height))
frame_counter = 0



vertices1 = np.array([[0,490],[482,300],[644,300],[540,720],[0,720]],np.int32)
pts_mask1 = vertices1.reshape((-1,1,2))

vertices2 = np.array([[764,720],[673,300],[773,300],[1255,720]],np.int32)
pts_mask2 = vertices2.reshape((-1,1,2))

black_frame = np.zeros((frame_height,frame_width,3), np.uint8)


# # Setup the termination criteria, either 10 iteration or move by atleast 1 pt
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1, 1 )

FLAG = False

cars_left_side = 0
cars_right_side = 0

while True:
    ret, frame = cap.read()
    
    if not ret: #if vid finish repeat
        frame = cv2.VideoCapture('overpass.mp4') 
        continue

    if ret == True:

        frame_orig = frame.copy()
        
        cv2.fillPoly(black_frame,[pts_mask1],color=(255,255,255))
        cv2.fillPoly(black_frame,[pts_mask2],color=(255,255,255))
        
        frame = cv2.bitwise_and(frame, black_frame)

        # gaussian blur
        frame_blur = cv2.GaussianBlur(frame,(5,5),10)
        
        # apply background subtractor
        frame_bg_subtractor = subtractor.apply(frame_blur)
            
        
        # apply a closing kernel followed by binary threshold
        frame_threshold = cv2.morphologyEx(frame_bg_subtractor, cv2.MORPH_CLOSE, kernel)
        
        # apply a opening kernel 
        frame_opening = cv2.morphologyEx(frame_threshold, cv2.MORPH_OPEN, kernel)
        
        # apply a dilate kernel 
        dilation = cv2.dilate(frame_opening, kernel)
        
        #draw a line on left lane
        cv2.line(frame_orig, (70,555), (540,555), (0,60,255), 5)
        
        #draw a line on right lane
        cv2.line(frame_orig, (703,464), (937,464), (255,127,0), 5)
        
        # ret,frame_opening = cv2.threshold(frame_opening,20,255,cv2.THRESH_BINARY)
        
        # #find countours in image
        contours2, hierarchy = cv2.findContours(frame_opening, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        external_contours = np.zeros(frame.shape)
        
        for i, element in enumerate(contours2):
            cv2.drawContours(external_contours, contours2, i , (0,255,0))
            for element_ in element:
                if(element_[0][0] < 675 and element_[0][1] > 0 and element_[0][1] < 234 and FLAG == False):

                    x, y, w, h = element_[0][0], element_[0][1], 70, 70 # simply hardcoded the values
                    track_window = (x, y, w, h)
                          
                    FLAG = True
    
        frame_counter += 1
        
        
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
                    cv2.drawMarker(frame_orig, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, markerSize=8, thickness=3,line_type=cv2.LINE_8)
                    
                    # left lane line (70,600), (540,600)
                    if((70 < cx < 540) and ( 550 < cy < 565)):
                        cars_left_side += 1
                        
                    # right lane line (703,464), (927,464)
                    if((cx > 703 and cx < 967) and (cy > 460 and cy < 466)):
                        cars_right_side += 1
        
        
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame_orig,text=str(cars_left_side),org=(100,500), fontFace=font,fontScale= 4,color=(0,0,255),thickness=8,lineType=cv2.LINE_AA)
         
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame_orig,text=str(cars_right_side),org=(1000,500), fontFace=font,fontScale= 4,color=(255,0,0),thickness=8,lineType=cv2.LINE_AA)
        
        
        
        # cv2.namedWindow('video with mask',cv2.WINDOW_NORMAL)
        # cv2.namedWindow('frame with blur',cv2.WINDOW_NORMAL)
        cv2.namedWindow('frame_orig',cv2.WINDOW_NORMAL)
        # cv2.namedWindow('frame_bg_subtractor',cv2.WINDOW_NORMAL)
        # cv2.namedWindow('frame_opening',cv2.WINDOW_NORMAL)        
        cv2.namedWindow('dilation',cv2.WINDOW_NORMAL)
        
        cv2.namedWindow('external_contours',cv2.WINDOW_NORMAL)
        # cv2.imshow('video with mask',frame)
        cv2.imshow('frame_orig',frame_orig)
        # cv2.imshow('frame with blur',frame_blur)
        # cv2.imshow('frame_bg_subtractor',frame_bg_subtractor)
        # cv2.imshow('frame_opening',frame_opening)
        cv2.imshow('dilation',dilation)
        cv2.imshow('external_contours',external_contours)
        
        ## save video
        out.write(frame_orig)        
        
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break
cap.release()
#out.release()
cv2.destroyAllWindows()


