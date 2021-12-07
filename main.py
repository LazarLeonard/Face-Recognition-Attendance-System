import cv2
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import *
from PIL import Image
import os
import csv
import datetime
import time

def is_number(val):
    try:
        float(val)
        return True
    except ValueError:
        pass
    try:
        import unicodedata
        unicodedata.numeric(val)
        return True
    except (TypeError, ValueError):
        pass 
    return False



def TakeImages():       
    Id=(txt.get())
    name=(txt2.get())
    if(is_number(Id) and name.isalpha()):
        cam = cv2.VideoCapture(0)
        harcascadePath = "haarcascade_frontalface_default.xml"
        detector=cv2.CascadeClassifier(harcascadePath)
        sampleNum=0
        while(True):
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faceSamples = detector.detectMultiScale(gray, 1.3, 5)
            for (x,y,w,h) in faceSamples:
                cv2.rectangle(img,(x,y),(x+w,y+h),(0, 255, 0), 3)        
                sampleNum=sampleNum+1
                cv2.imwrite("Dataset\ "+name +"."+Id +'.'+ str(sampleNum) + ".jpg", gray[y:y+h,x:x+w])
                cv2.imshow('frame',img)
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
            elif sampleNum>40:
                break
        cam.release()
        cv2.destroyAllWindows() 
        res = "Imaginile au fost salvate pentru \n ID = " + Id +" Nume = "+ name
        row = [Id , name]
        with open('DetaliiPersoane\DetaliiPersoane.csv','a+') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(row)
        csvFile.close()
        msg.configure(text= res)
    else:
        if(is_number(Id)):
            res = "Introduceti un nume alfabetic"
            msg.configure(text= res)
        if(name.isalpha()):
            res = "Introduceti un ID numeric"
            msg.configure(text= res)
    
def TrainImages():
    recognizer = cv2.face_LBPHFaceRecognizer.create()
    harcascadePath = "haarcascade_frontalface_default.xml"
    detector =cv2.CascadeClassifier(harcascadePath)
    faceSamples,Id = getImagesAndLabels("Dataset")
    recognizer.train(faceSamples, np.array(Id))
    recognizer.save("TrainingImageLabel\Trainner.yml")
    res = "Dataset Antrenat!"
    msg.configure(text= res)

def getImagesAndLabels(path):
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)] 
    faceSamples=[]
    Ids=[]
    for imagePath in imagePaths:
        image_pil=Image.open(imagePath).convert('L')
        imageNp=np.array(image_pil,'uint8')
        Id=int(os.path.split(imagePath)[-1].split(".")[1])
        faceSamples.append(imageNp)
        Ids.append(Id)        
    return faceSamples,Ids

def TrackImages():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("TrainingImageLabel\Trainner.yml") 
    harcascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(harcascadePath); 
    df=pd.read_csv("DetaliiPersoane\DetaliiPersoane.csv")
    cam = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX      
    col_names =  ['Id','Name','Date','Time']
    prezenta = pd.DataFrame(columns = col_names)    
    while True:
        ret, im =cam.read()
        gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        faceSamples=faceCascade.detectMultiScale(gray, 1.2,5)    
        for(x,y,w,h) in faceSamples:
            cv2.rectangle(im,(x,y),(x+w,y+h),(225,0,0),2)
            Id, conf = recognizer.predict(gray[y:y+h,x:x+w])
            cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 3)                                 
            if(conf < 45):
                ts = time.time()      
                date = datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%y')
                timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                aa=df.loc[df['Id'] == Id]['Name'].values
                tt=str(Id)+"-"+aa
                prezenta.loc[len(prezenta)] = [Id,aa,date,timeStamp]
                
            else:
                Id='Necunoscut'                
                tt=str(Id)
            if(conf > 75):
                noOfFile=len(os.listdir("PersoaneNecunoscute"))+1
                cv2.imwrite("PersoaneNecunoscute\Image"+str(noOfFile) + ".jpg", im[y:y+h,x:x+w])            
            cv2.putText(im,str(tt),(x,y+h), font, 1,(255,255,255),2)        
        prezenta=prezenta.drop_duplicates(subset=['Id'],keep='first')    
        cv2.imshow('im',im) 
        if (cv2.waitKey(1)==ord('q')):
            break
    ts = time.time()    
    date = datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%y')
    timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
    Hour,Minute,Second=timeStamp.split(":")
    fileName="Prezenta\Prezenta_"+date+"_"+Hour+"-"+Minute+"-"+Second+".csv"
    prezenta.to_csv(fileName,index=False)
    cam.release()
    cv2.destroyAllWindows()
    res=prezenta
    msg2.configure(text= res)


window = tk.Tk()
window.title("Recunoastere_Faciala")

dialog_title = 'Iesi'

window.configure(background='sky blue')


window.grid_rowconfigure(0, weight=1)
window.grid_columnconfigure(0, weight=1)




msg = tk.Label(window, text="Identificarea in timp real dintr-un stream live a unor persoane" ,bg="sky blue"  ,fg="white"  ,width=50  ,height=3,font=('times', 30, 'italic bold underline')) 

msg.place(x=200, y=20)

label = tk.Label(window, text="Introduceti ID",width=20  ,height=2  ,fg="sky blue"  ,bg="white" ,font=('times', 15, ' bold ') ) 
label.place(x=400, y=200)

txt = tk.Entry(window,width=25  ,bg="white" ,fg="sky blue",font=('times', 15, ' bold '))
txt.place(x=700, y=215)

labeltwo = tk.Label(window, text="Introduceti Nume",width=20  ,fg="sky blue"  ,bg="white" ,height=2 ,font=('times', 15, ' bold ')) 
labeltwo.place(x=400, y=300)

txt2 = tk.Entry(window,width=25  ,bg="white"  ,fg="sky blue",font=('times', 15, ' bold ')  )
txt2.place(x=700, y=315)

labelthree = tk.Label(window, text="Notificari : ",width=20  ,fg="sky blue"  ,bg="white"  ,height=2 ,font=('times', 15, ' bold underline ')) 
labelthree.place(x=400, y=400)

msg = tk.Label(window, text="" ,bg="white"  ,fg="sky blue"  ,width=22  ,height=2, activebackground = "white" ,font=('times', 15, ' bold ')) 
msg.place(x=700, y=400)

labelthreee = tk.Label(window, text="Prezenta : ",width=20  ,fg="sky blue"  ,bg="white"  ,height=3 ,font=('times', 15, ' bold  underline')) 
labelthreee.place(x=400, y=650)


msg2 = tk.Label(window, text="" ,fg="sky blue"   ,bg="white",activeforeground = "green",width=30  ,height=3  ,font=('times', 15, ' bold ')) 
msg2.place(x=700, y=650)

takeImg = tk.Button(window, text="Extrage setul de date", command=TakeImages  ,fg="sky blue"  ,bg="white"  ,width=20  ,height=3, activebackground = "sky blue" ,font=('times', 15, ' bold '))
takeImg.place(x=200, y=500)
trainImg = tk.Button(window, text="Antreneaza setul de date", command=TrainImages  ,fg="sky blue"  ,bg="white"  ,width=20  ,height=3, activebackground = "sky blue" ,font=('times', 15, ' bold '))
trainImg.place(x=500, y=500)
trackImg = tk.Button(window, text="Identifica persoana", command=TrackImages  ,fg="sky blue"  ,bg="white"  ,width=20  ,height=3, activebackground = "sky blue" ,font=('times', 15, ' bold '))
trackImg.place(x=800, y=500)
quitWindow = tk.Button(window, text="Iesi", command=window.destroy  ,fg="sky blue"  ,bg="white"  ,width=20  ,height=3, activebackground = "sky blue" ,font=('times', 15, ' bold '))
quitWindow.place(x=1100, y=500)



window.mainloop()