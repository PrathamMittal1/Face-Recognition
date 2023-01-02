import csv
import tkinter as tk
import cv2
import pandas as pd
from extras import *

init_info()
window = tk.Tk()
window.title('Face Recognition')
window.attributes('-fullscreen',True)
window.configure(background='#ffffff')
window.grid_rowconfigure(0, weight=1)
window.grid_columnconfigure(0, weight=1)
message = tk.Label(window, text="Face-Recognition-System", bg="blue", fg="white", width=50,
                   height=3, font=('comic sans ms', 30, 'bold'))
message.place(x=200, y=20)
lb1 = tk.Label(window, text='No.', width=20, height=2,
               fg='blue', bg='#ffffff', font=('times', 15, 'bold'))
lb1.place(x=400, y=200)
txt = tk.Entry(window, width=20, bg='#ffffff', fg='#0000bb',
               font=('times', 15, 'bold'))
txt.place(x=700, y=215)
lb2 = tk.Label(window, text='Name', width=20, height=2,
               fg='blue', bg='#ffffff', font=('times', 15, 'bold'))
lb2.place(x=400, y=300)
txt2 = tk.Entry(window, width=20, bg='#ffffff', fg='#0000bb',
                font=('times', 15, 'bold'))
txt2.place(x=700, y=315)




def TakeImages():
    Id, name = txt.get(), txt2.get()
    if is_number(Id) and name.isalpha():
        cam = cv2.VideoCapture(0)
        detector = cv2.CascadeClassifier(r"data\haarcascade_frontalface_default.xml")
        sampleNum = 0
        while 1:
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray)
            for x, y, w, h in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                sampleNum += 1
                cv2.imwrite("TrainingImage\\" + name + "." + Id + "." + str(sampleNum) + ".jpg",
                            gray[y:y + h, x:x + w])
                cv2.imshow('frame', img)
            if cv2.waitKey(1) == 27:
                break
            elif sampleNum > 60:
                break
        cam.release()
        cv2.destroyAllWindows()
        res = 'Images saved for ID: ' + Id + ' Name: ' + name
        row = [Id, name]
        with open(r'UserDetails\UserDetails.csv', 'a+') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(row)
        csvFile.close()
        message.configure(text=res)
    else:
        if (is_number(Id)):
            res = "Enter Alphabetical Name"
            message.configure(text=res)
        if name.isalpha():
            res = 'Enter Numeric Id'
            message.configure(text=res)


def TrainImages():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    faces, Id = getImagesAndLabels('TrainingImage')
    recognizer.train(faces, np.array(Id))
    recognizer.save(r"TrainingImageLabel\Trainner.yml")
    res = 'Image Trained'
    message.configure(text=res)





def TrackImages():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("TrainingImageLabel\\Trainner.yml")
    faceCascade = cv2.CascadeClassifier(r"data\haarcascade_frontalface_default.xml")
    df = pd.read_csv(r"UserDetails\UserDetails.csv")
    cam = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    while True:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (225, 0, 0), 2)
            Id, conf = recognizer.predict(gray[y:y + h, x:x + w])
            if conf < 50:
                aa = df.loc[df['Id'] == Id]['Name'].values
                tt = str(Id) + '-' +aa
            else:
                Id = 'Unknown'
                tt = str(Id)
            if conf > 75:
                noOfFile = len(os.listdir("ImagesUnknown")) + 1
                cv2.imwrite("ImagesUnknown\\Image" + str(noOfFile) + ".jpg", img[y:y + h, x:x + w])
            cv2.putText(img, str(tt), (x, y + h + 25), font, 1, (50, 200, 235), 2)
        cv2.imshow('img', img)
        if cv2.waitKey(1) == 27:
            break
    cam.release()
    cv2.destroyAllWindows()


takeImg = tk.Button(window, text="Sample",command=TakeImages, fg="white", bg='blue',
                    width=20, height=3, activebackground='#ff8800',font=('times', 15, ' bold '))
takeImg.place(x=200, y=500)
trainImg = tk.Button(window, text="Train Model",command=TrainImages, fg="white", bg='blue',
                     width=20, height=3, activebackground='#ff8800',font=('times', 15, ' bold '))
trainImg.place(x=500, y=500)
trackImg = tk.Button(window, text="Testing",command=TrackImages, fg="white", bg='blue',
                     width=20, height=3, activebackground='#ff8800',font=('times', 15, ' bold '))
trackImg.place(x=800, y=500)
quitWindow = tk.Button(window, text="Quit",command=window.destroy, fg="white", bg='blue',
                       width=20, height=3, activebackground='#ff8800',font=('times', 15, ' bold '))
quitWindow.place(x=1100, y=500)

window.mainloop()

