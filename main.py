import cv2
import imutils
import numpy as np
import pytesseract
import sqlite3
pytesseract.pytesseract.tesseract_cmd = r'D:\Tesseract-OCR\tesseract.exe'

webcam = cv2.VideoCapture(0) 
cam=cv2.VideoCapture(0)
def findColor(path):
    while(1): 
	    #ret,imageFrame = webcam.read()
	    #cv2.imshow('frame', imageFrame)
	    imageFrame=cv2.imread(path,cv2.IMREAD_COLOR)
	    hsvFrame = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV) 

	
	    red_lower = np.array([136, 87, 111], np.uint8) 
	    red_upper = np.array([180, 255, 255], np.uint8) 
	    red_mask = cv2.inRange(hsvFrame, red_lower, red_upper) 

	
	
	    green_lower = np.array([25, 52, 72], np.uint8) 
	    green_upper = np.array([102, 255, 255], np.uint8) 
	    green_mask = cv2.inRange(hsvFrame, green_lower, green_upper) 

	

	    blue_lower = np.array([94, 80, 2], np.uint8) 
	    blue_upper = np.array([120, 255, 255], np.uint8) 
	    blue_mask = cv2.inRange(hsvFrame, blue_lower, blue_upper) 
	
	
	

	
	    kernal = np.ones((5, 5), "uint8") 
	
	
	    red_mask = cv2.dilate(red_mask, kernal) 
	    res_red = cv2.bitwise_and(imageFrame, imageFrame,mask = red_mask) 
	
	
	    green_mask = cv2.dilate(green_mask, kernal) 
	    res_green = cv2.bitwise_and(imageFrame, imageFrame,	mask = green_mask) 
	
	 
	    blue_mask = cv2.dilate(blue_mask, kernal) 
	    res_blue = cv2.bitwise_and(imageFrame, imageFrame,mask = blue_mask) 

	
	    contours, hierarchy = cv2.findContours(red_mask,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
	
	    for pic, contour in enumerate(contours): 
		    area = cv2.contourArea(contour) 
		    if(area > 300): 
			    x, y, w, h = cv2.boundingRect(contour) 
			    imageFrame = cv2.rectangle(imageFrame, (x, y),(x + w, y + h), (0, 0, 255), 2) 
			
			    cv2.putText(imageFrame, "Red Colour", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255))	 

	
	    contours, hierarchy = cv2.findContours(green_mask,	cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) 
	
	    for pic, contour in enumerate(contours): 
		    area = cv2.contourArea(contour) 
		    if(area > 300): 
			    x, y, w, h = cv2.boundingRect(contour) 
			    imageFrame = cv2.rectangle(imageFrame, (x, y),(x + w, y + h), (0, 255, 0), 2) 
			
			    cv2.putText(imageFrame, "Green Colour", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0)) 

	
	    contours, hierarchy = cv2.findContours(blue_mask,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
	    for pic, contour in enumerate(contours): 
		    area = cv2.contourArea(contour) 
		    if(area > 300): 
			    x, y, w, h = cv2.boundingRect(contour) 
			    imageFrame = cv2.rectangle(imageFrame, (x, y),(x + w, y + h), (255, 0, 0), 2) 
			
			    cv2.putText(imageFrame, "Blue Colour", (x, y),cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0)) 
			
	
	    cv2.imshow("Multiple Color Detection in Real-TIme", imageFrame) 
	    if cv2.waitKey(10) & 0xFF == ord('q'): 
		    #cap.release() 
		    cv2.destroyAllWindows() 
		    break



def findPlate(path):
    img = cv2.imread(path,cv2.IMREAD_COLOR)
    print(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    gray = cv2.bilateralFilter(gray, 13, 15, 15) 

    edged = cv2.Canny(gray, 30, 200) 
    contours = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)[:10]
    screenCnt = None

    for c in contours:
    
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.018 * peri, True)
 
        if len(approx) == 4:
            screenCnt = approx
            break

    if screenCnt is None:
        detected = 0
        print ("No contour detected")
    else:
        detected = 1

    if detected == 1:
        cv2.drawContours(img, [screenCnt], -1, (0, 255, 0), 3)

    mask = np.zeros(gray.shape,np.uint8)
    new_image = cv2.drawContours(mask,[screenCnt],0,255,-1,)
    new_image = cv2.bitwise_and(img,img,mask=mask)

    (x, y) = np.where(mask == 255)
    (topx, topy) = (np.min(x), np.min(y))
    (bottomx, bottomy) = (np.max(x), np.max(y))
    Cropped = gray[topx:bottomx+1, topy:bottomy+1]

    text = pytesseract.image_to_string(Cropped, config='--psm 11')
    print("License Plate Recognition\n")
    print("Detected license plate Number is:",text)
    img = cv2.resize(img,(500,300))
    Cropped = cv2.resize(Cropped,(400,200))
    cv2.imshow('car',img)
    cv2.imshow('Cropped',Cropped)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return text
  

def database():
    connection=sqlite3.connect('hackathon.db')
    print(" DataBase Connected sucessfully")
   # connection.execute('''create table theftVehicle(id int auto_increment,username varchar(255),color varchar(255),plate varchar(255),blackORstolen varchar(255) ,primary key(id));''')
   # print ("table sucessfully")

    


def find(text):
    connection=sqlite3.connect('hackathon.db')
    connection.execute('''INSERT INTO theftVehicle(username,color,plate,blackORstolen) VALUES('ramakrishna','blue','DZI7YXR','stolen');''')
    connection.execute('''INSERT INTO theftVehicle(username,color,plate,blackORstolen) VALUES('abilash','black','DL7CQ1939','blacklisted');''')
    connection.execute('''INSERT INTO theftVehicle(username,color,plate,blackORstolen) VALUES('nirmal','blue','FT856VD','stolen');''')
    connection.execute('''INSERT INTO theftVehicle(username,color,plate,blackORstolen) VALUES('balaji','white','TN21BZ0768','blacklisted');''')
    connection.execute('''INSERT INTO theftVehicle(username,color,plate,blackORstolen) VALUES('arun','brown','MH01TMP8145','stolen');''')
    connection.execute('''INSERT INTO theftVehicle(username,color,plate,blackORstolen) VALUES('rakesh','red','TN09BV6196','blacklisted');''')
    connection.execute('''INSERT INTO theftVehicle(username,color,plate,blackORstolen) VALUES('hari','red','WB24AK0333','stolen');''')
    connection.execute('''INSERT INTO theftVehicle(username,color,plate,blackORstolen) VALUES('peter','black','TN09BU1357','blacklisted');''')
    connection.execute('''INSERT INTO theftVehicle(username,color,plate,blackORstolen) VALUES('krish','white','TN22DK3510','stolen');''')
    connection.execute('''INSERT INTO theftVehicle(username,color,plate,blackORstolen) VALUES('karthik','violet','RJ14CE5678','blacklisted');''')
    connection.execute('''INSERT INTO theftVehicle(username,color,plate,blackORstolen) VALUES('atommax','brown','MH46N4832','stolen');''')
    print("Sample Data Are loaded")
    
    text=text.replace(" ","").upper()
    print(text)
    cursor=connection.execute("select username,color,blackORstolen from theftVehicle where plate='"+text+"' ")
    print("\n")
    
    print("RESULT",*cursor)
   
    print("\n")




path='download.jpg'
plate=findPlate(path)
findColor(path)
#database()
find(plate)