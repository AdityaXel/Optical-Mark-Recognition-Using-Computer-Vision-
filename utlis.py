import cv2
import numpy as np

def concatenate_images(imgArray, scale=1.0, labels=None):
    """
    Concatenates images from an array in a grid format with optional scaling.
    Displays a label with white background and moderate text size on each image.

    Parameters:
        imgArray: List of images (NumPy arrays). Can be a single list for one row,
                  or a list of lists for multiple rows.
        scale: Float, scaling factor for resizing each image (default is 1.0).
        labels: List of strings (for 1D imgArray) or list of list of strings (for 2D imgArray).
    """
    is_2d = isinstance(imgArray[0], list)

    def add_label(img, text):
        """Draw white rectangle and text label on image."""
        if text:
            font_scale = 0.7  # moderate readable size
            thickness = 2
            font = cv2.FONT_HERSHEY_SIMPLEX
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
            text_x, text_y = 10, 30
            rect_w = text_size[0] + 20
            rect_h = text_size[1] + 10

            # Draw white background rectangle
            cv2.rectangle(img, (text_x - 5, text_y - text_size[1] - 5),
                          (text_x - 5 + rect_w, text_y + 5), (255, 255, 255), -1)

            # Draw text (green or black for contrast)
            cv2.putText(img, text, (text_x, text_y),
                        font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
        return img

    if is_2d:
        rows = len(imgArray)
        cols = len(imgArray[0])
        width = int(imgArray[0][0].shape[1] * scale)
        height = int(imgArray[0][0].shape[0] * scale)

        for row in range(rows):
            for col in range(cols):
                img = imgArray[row][col]
                imgArray[row][col] = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
                if len(imgArray[row][col].shape) == 2:
                    imgArray[row][col] = cv2.cvtColor(imgArray[row][col], cv2.COLOR_GRAY2BGR)

                # Add label
                if labels is not None:
                    try:
                        text = labels[row][col]
                    except Exception:
                        text = ""
                    imgArray[row][col] = add_label(imgArray[row][col], text)

        hor = [np.hstack(imgArray[row]) for row in range(rows)]
        result = np.vstack(hor)

    else:
        cols = len(imgArray)
        width = int(imgArray[0].shape[1] * scale)
        height = int(imgArray[0].shape[0] * scale)

        for i in range(cols):
            img = imgArray[i]
            imgArray[i] = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
            if len(imgArray[i].shape) == 2:
                imgArray[i] = cv2.cvtColor(imgArray[i], cv2.COLOR_GRAY2BGR)

            # Add label
            if labels is not None:
                try:
                    text = labels[i]
                except Exception:
                    text = ""
                imgArray[i] = add_label(imgArray[i], text)

        result = np.hstack(imgArray)

    return result



def rectcontour(contours):
    
    rectCon = []
    for i in contours :
        area = cv2.contourArea(i)
        #print("Area",area)
        if area>50 :
            peri = cv2.arcLength(i,True)
            approx = cv2.approxPolyDP(i, 0.02*peri, True)
            print("Corner Points", approx)
            if len(approx) == 4 :
                rectCon.append(i)
    rectCon = sorted(rectCon, key =cv2.contourArea, reverse= True)
    
    return rectCon

def getCornerPoints(cont):
    peri = cv2.arcLength(cont ,True)
    approx = cv2.approxPolyDP(cont, 0.02*peri, True)
    return approx 

def reorder(myPoints):
    
    myPoints = myPoints.reshape((4,2))
    myPointsNew = np.zeros((4,1,2), np.int32) 
    add = myPoints.sum(1) 
    #print(myPoints)
    #print(add)
    myPointsNew[0] = myPoints[np.argmin(add)] # top left (0,0)
    myPointsNew[3] = myPoints[np.argmax(add)] # bottom right (w,h)
    diff = np.diff(myPoints, axis = 1)
    myPointsNew[1] = myPoints[np.argmin(diff)] # [w,0] # top right
    myPointsNew[2] = myPoints[np.argmax(diff)] # [0,h] # bottom left
    # print(diff)
    
    return myPointsNew

def splitBoxes(img) :
    rows = np.vsplit(img,5)  
    boxes = [ ]
    
    for r in rows :
        cols = np.hsplit(r,5)
        
        for box in cols :
            boxes.append(box)
            #cv2.imshow("Split", box) 
    return boxes 

     
def showAnswers(img, myIndex, grading, ans, questions, choices) :
    secW = int(img.shape[1]/questions)
    secH = int(img.shape[0]/questions)
    
    for x in range(0, questions) :
        myAns = myIndex[x]
        Cx = (myAns*secW) + secW // 2
        cY = (x*secH) + secH //2    
        
        if grading[x] == 1 :
            myColor = (0,255,0)
        else :
            myColor = (0,0,255)
            correctAns = ans[x]
            cv2.circle(img,((correctAns*secW)+secW//2,(x*secH)+ secH//2), 20,(0,255,0), cv2.FILLED)
             
        cv2.circle(img,(Cx,cY), 50, myColor, cv2.FILLED)  
        
    return img 
          
    
    
    
    
    