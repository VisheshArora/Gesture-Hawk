import cv2
import numpy as np
import serial
ser = serial.Serial('/dev/ttyACM0', baudrate = 9600)
cap = cv2.VideoCapture(0)                                           #for capturing video frame by frame
cp = cv2.imread('stop.jpg')                                           #for reading image with ist left contour
hsp = cv2.cvtColor(cp, cv2.COLOR_BGR2HSV)
lower_blue = np.array([110,50,50])
upper_blue = np.array([130,255,255])

mask1 = cv2.inRange(hsp, lower_blue, upper_blue)                             #blue color filter mask
res1 = cv2.bitwise_and(cp,cp, mask= mask1)

imgrey1 = cv2.cvtColor(res1,cv2.COLOR_BGR2GRAY)
ret1,thresh1 = cv2.threshold(imgrey1,127,255,cv2.THRESH_BINARY_INV)
_,contours, hierar = cv2.findContours(imgrey1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)



sz = len(contours)
arr = np.arange(sz).reshape((sz,1))
for i in range(0,sz):
 arr[i] = cv2.contourArea(contours[i])
maxAr0 = np.amax(arr)
cin = np.argmax(arr)
conStop = contours[cin]
cp = cv2.drawContours(cp,contours,cin, (0,255,0), 3)
cv2.imshow("StopContour",cp)




cp1 = cv2.imread('for1.jpg')
hsa = cv2.cvtColor(cp1, cv2.COLOR_BGR2HSV)
mask2 = cv2.inRange(hsa, lower_blue, upper_blue)
res2 = cv2.bitwise_and(cp1,cp1, mask= mask2)
imgrey2 = cv2.cvtColor(res2,cv2.COLOR_BGR2GRAY)
ret2,thresh2 = cv2.threshold(imgrey2,127,255,cv2.THRESH_BINARY_INV)
_,conto, hierarc = cv2.findContours(imgrey2,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
sz1 = len(conto)
ar1 = np.arange(sz1).reshape((sz1,1))
for i in range(0,sz1):
 ar1[i] = cv2.contourArea(conto[i])
maxAr1 = np.amax(ar1)
cindex1 = np.argmax(ar1)
conFor = conto[cindex1]
cp1 = cv2.drawContours(cp1,conto,cindex1, (0,255,0), 3)
cv2.imshow("contour2",cp1)
#cv2.imshow("Standard",cp)

while(1):
    signal = 'N'
    _, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    lower_red = np.array([110,50,50])
    upper_red = np.array([130,255,255])
    
    mask = cv2.inRange(hsv, lower_red, upper_red)
    res = cv2.bitwise_and(frame,frame, mask= mask)
    imgrey = cv2.cvtColor(res,cv2.COLOR_BGR2GRAY)
    _,grey = cv2.threshold(imgrey,98,200,cv2.THRESH_BINARY_INV)
    th3 = cv2.adaptiveThreshold(imgrey,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    
    _,cont, hierarchy = cv2.findContours(imgrey,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    
    sz = len(cont)
    ar = np.arange(sz).reshape((sz,1))
    for i in range(0,sz):
     ar[i] = cv2.contourArea(cont[i])
    maxAr = np.amax(ar)
    cindex = np.argmax(ar)
    cn2 = cont[cindex]
    handArea = cv2.contourArea(cont[cindex]) 
    #frame = cv2.drawContours(frame,cont,cindex,(0,255,0),2)
    x,y,w,h = cv2.boundingRect(cont[cindex])



    extRight = tuple(cn2[cn2[:, :, 0].argmax()][0])
    extLeft = tuple(cn2[cn2[:, :, 0].argmin()][0])
    topMost = tuple(cn2[cn2[:,:,1].argmin()][0])
    botMost = tuple(cn2[cn2[:,:,1].argmax()][0])
    if handArea > 13000:
      cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
      cv2.circle(frame, extRight, 8, (0,255,0), -1)
      cv2.circle(frame, extLeft, 8, (0,255,0), -1)
      cv2.circle(frame, topMost, 8, (0,255,0), -1)
    hull = cv2.convexHull(cn2,returnPoints = False)
    defects = cv2.convexityDefects(cn2,hull)
    if handArea > 13000:
     defShape = defects.shape[0]
     dis = np.arange(defShape).reshape((defShape,1))
     for i in range(defects.shape[0]):
      s,e,f,d = defects[i,0]
      start = tuple(cn2[s][0])
      end = tuple(cn2[e][0])
      far = tuple(cn2[f][0])
      dis[i] = -1*((far[0]-start[0])*(start[1]-end[1])-(far[1]-start[1])*(start[0]-end[0]))/(((start[1]-end[1])**2 + (start[0] - end[0])**2)**(0.5))
      cv2.line(cp1,start,end,[0,255,0],2)
      cv2.line(cp1,far,end,[0,255,0],2)
      cv2.line(cp1,far,start,[0,255,0],2)
      cv2.circle(cp1,far,5,[0,0,255],-1)
    #print(handArea)
    #print(compFrac)
     maxLine_ind = np.argmax(dis)
     a,d,b,w = defects[maxLine_ind,0]
     farEnd = tuple(cn2[b][0]) 
     cv2.circle(frame,farEnd,5,[255,0,0],-1)
     distRF = ((extRight[1]-farEnd[1])**2 + (extRight[0]-farEnd[0])**2)**(0.5)
     distLF = ((extLeft[1]-farEnd[1])**2 + (extLeft[0]-farEnd[0])**2)**(0.5)
     distMax = -1*(topMost[1] - extLeft[1])
     distMax1 = -1*(topMost[1]-botMost[1])
     compFrac = cv2.matchShapes(cont[cindex],conStop,1,0.0)                               #MatchShape for Stop Signal
     compFrac1 = cv2.matchShapes(cont[cindex],conFor,1,0.0)                               #MatchShape for Forward Signal
     #cv2.line(frame,extLeft,topMost,[0,255,0],2)
     cv2.line(frame,botMost,topMost,[0,255,0],2)
     print(distMax1)
     lR = float(0.019)
     hR = float(0.150)
     compFrac = float(compFrac)
     compFrac1 = float(compFrac1)
     if (distRF<distLF and not(compFrac1 >= lR and compFrac1 <= hR) and handArea > 13000 and not(distMax <= 210 and distMax >= 140) and not(distMax1 <= 120 and distMax1 >= 80)):
      slope = (-extRight[1]+farEnd[1])/(extRight[0]-farEnd[0])
      cv2.putText(frame,"LEFT",(10,100),cv2.FONT_HERSHEY_SIMPLEX,4,(0,0,255),2,cv2.LINE_AA)
      signal ='L'
      ser.write(b'L')
     if(distRF>distLF and not(compFrac1 >= lR and compFrac1 <= hR) and handArea > 13000 and not(distMax <= 210 and distMax >= 140) and not(distMax1 <= 120 and distMax1 >= 80)):
      slope = (-extLeft[1]+farEnd[1])/(extLeft[0]-farEnd[0])
      cv2.putText(frame,"RIGHT",(10,100),cv2.FONT_HERSHEY_SIMPLEX,4,(0,0,255),2,cv2.LINE_AA)
      signal ='R'
      ser.write(b'R')
    #print(compFrac1)
     if(distMax1 <= 120 and distMax1 >= 80 and handArea > 13000):
      cv2.putText(frame,"STOP",(10,100),cv2.FONT_HERSHEY_SIMPLEX,4,(0,0,255),2,cv2.LINE_AA)
      signal = 'S'
      ser.write(b'S')
     if(compFrac1 >= lR and compFrac1 <= hR and handArea > 13000):
     #if(distMax <= 100 and distMax >= 10 and handArea > 12000): 
      cv2.putText(frame,"FORWARD",(10,100),cv2.FONT_HERSHEY_SIMPLEX,4,(0,0,255),2,cv2.LINE_AA)
      signal = 'F'
      ser.write(b'F')
    #cv2.imshow("threshold",th3)
    #ser.write(signal)	
    cv2.imshow("Adaptive Thershold",th3)
    cv2.imshow("Mask",res)
    cv2.imshow("Contours",frame)
    
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
     break

cv2.destroyAllWindows()
cap.release()
ser.close()	    	
