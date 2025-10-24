import cv2        
import numpy as np      
import utlis     

##################################################################
path = "Computer Vision Project/Screenshot 2025-10-23 172739.png"
width = 700 
height = 700 
questions = 5
Choices = 5 
ans = [1,2,1,1,4]
webCamFeed = True
cameraNo = 0 # Default Camera
##################################################################
cap = cv2.VideoCapture(cameraNo)
cap.set(10,150) # Setting Brightness here 

while True :
    if webCamFeed : success, img = cap.read()
    else : img = cv2.imread(path)


# PREPROCESSING 
    img = cv2.resize(img, (width, height))
    imgcontours =  img.copy() # Copy the Image here 
    imgbiggestcontour = img.copy()
    imgFinal = img.copy()
    imgFinal = cv2.resize(imgFinal, (width, height))
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(img,(5,5), 1)  
    imgcanny = cv2.Canny(imgBlur, 10, 50)
    
    
    try :
        # FINDING ALL CONTOURS 
        contours, heirarchy = cv2.findContours(imgcanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(imgcontours, contours, -1, (0,255,0), 10)

        # FIND THE RECTANGLES 
        rectCon = utlis.rectcontour(contours)
        biggestContour = utlis.getCornerPoints(rectCon[0])
        print(biggestContour.shape)
        gradePoints = utlis.getCornerPoints(rectCon[1])
        # print(biggestContour)

        if biggestContour.size != 0 and gradePoints.size !=0 :
            
            cv2.drawContours(imgbiggestcontour, biggestContour, -1, (0,255,0), 10) 
            cv2.drawContours(imgbiggestcontour, gradePoints, -1, (255,0,0),10) 
            biggestContour = utlis.reorder(biggestContour)
            gradePoints = utlis.reorder(gradePoints)  
            
            pt1 = np.float32(biggestContour)
            pt2 = np.float32([[0,0],[width,0], [0,height], [width, height]]) 
            matrix = cv2.getPerspectiveTransform(pt1, pt2)
            imgWarpColored = cv2.warpPerspective(img, matrix, (width, height))
            
            ptG1 = np.float32(gradePoints)
            ptG2 = np.float32([[0,0],[325,0], [0,150], [325, 150]]) 
            matrixG = cv2.getPerspectiveTransform(ptG1, ptG2)
            imgGradeDisplay = cv2.warpPerspective(img, matrixG, (325, 150))
            #cv2.imshow("Grade", imgGradeDisplay)
            
            # APPLY THRESHOLD FOR CHECKING ANSWERS
            imgWarpGray = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)
            imgThresh = cv2.threshold(imgWarpGray,170,250, cv2.THRESH_BINARY_INV)[1] 
            
            boxes = utlis.splitBoxes(imgThresh)
            cv2.imshow("Test", boxes[2])
            #print(cv2.countNonZero(boxes[1]), cv2.countNonZero(boxes[2]) )
            
            # GETTING NON ZERO PIXEL VALUES : 
            myPixelVal = np.zeros((questions, Choices)) 
            countC = 0 
            countR = 0 
            
            for image in boxes :
                total_pixels = cv2.countNonZero(image)
                myPixelVal[countR][countC] = total_pixels
                countC += 1 
                if(countC == Choices) : countR +=1 ; countC = 0 
            #print(myPixelVal) 
            
            
            # FINDING INDEX VALUES OF THE MARKING 
            myIndex = [ ]
            for x in range(0, questions) : 
                arr = myPixelVal[x]
                #print("arr", arr)
                myIndexVal = np.where(arr == np.amax(arr)) # Gives maximum value here .
                #print(myIndexVal[0])
                myIndex.append(myIndexVal[0][0])
                myIndex = [int(x) for x in myIndex]
            print(myIndex) 
            
            # GRADING 
            grading = [ ]
            for x in range(0, questions):
                if ans[x] == myIndex[x] :
                    grading.append(1)
                else :
                    grading.append(0)
            print(grading)
                    
            score = (sum(grading)/ questions) * 100 # FINAL GRADE 
            print(score)   
            
            # Displaying Answer 
            imgResult = imgWarpColored.copy()
            utlis.showAnswers(imgResult, myIndex, grading, ans, questions, Choices)    
            imRawDrawing = np.zeros_like(imgWarpColored)    
            imRawDrawing = utlis.showAnswers(imRawDrawing,myIndex, grading, ans, questions, Choices)  
            invMatrix = cv2.getPerspectiveTransform(pt2, pt1)
            imgInvWarp = cv2.warpPerspective(imRawDrawing, invMatrix, (width, height))
            # Jase image phle dikh rhi thi vase hi answer ki image kr di 
            
            imgRawGrade = np.zeros_like(imgGradeDisplay)
            cv2.putText(imgRawGrade, str(int(score)) + "%", (50,100), cv2.FONT_HERSHEY_SIMPLEX,3,(0,255,255),3)  
            cv2.imshow("Grade", imgRawGrade)  
            invMatrixG = cv2.getPerspectiveTransform(ptG2, ptG1)
            imgInvGradeDisplay = cv2.warpPerspective(imgRawGrade, invMatrixG, (width, height))
            
            imgFinal = cv2.addWeighted(imgFinal, 1, imgInvWarp, 1, 0)
            imgFinal = cv2.addWeighted(imgFinal,1, imgInvGradeDisplay,1,0)
            
                

        imgBlank = np.zeros_like(img)  

        imageArray = ([img, imgGray, imgBlur, imgcanny ],
                    [imgcontours, imgbiggestcontour, imgWarpColored, imgThresh],
                    [imgResult, imRawDrawing, imgInvWarp , imgFinal])
    except :
        imgBlank = np.zeros_like(img)
        imageArray = ([img, imgGray, imgBlur, imgcanny ],
                    [imgBlank, imgBlank, imgBlank, imgBlank],
                    [imgBlank, imgBlank, imgBlank , imgBlank])

    labels = [["Original","Gray","Blurr","Canny"],["Contours","Biggest Cont","Warp","Threshold"],
            ["Result","Raw","InvWarp","Final"]]
    imgStacked = utlis.concatenate_images(imageArray, 0.2, labels)


    cv2.imshow("Final Image", imgFinal)
    cv2.imshow('Stacked Image', imgStacked)
    if cv2.waitKey(1) & 0xFF == ord("s") :
        cv2.imwrite("Final Result.jpg",imgFinal)
        cv2.waitKey(300)
     
         
            
    