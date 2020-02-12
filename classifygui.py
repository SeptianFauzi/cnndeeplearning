from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os
import matplotlib.pyplot as plt
from imutils import paths
import pandas as pd
import seaborn as sns
import matplotlib.patches as mpatches
from tkinter import *
from tkinter.ttk import *
from PIL import Image, ImageTk
from time import sleep

plt.rcParams.update({'font.size': 6})

size_datauji = 0

def proses_aja():
    global size_datauji
    model = load_model(input_model.get())
    lb = pickle.loads(open(input_label.get(), "rb").read())
    imagePaths = sorted(list(paths.list_images(input_data_uji.get())))
    size_datauji = len(imagePaths)
    no = 0
    import time 
    for imagePath in imagePaths:
# load the image
        print("[INFO] loading images...")
        image = cv2.imread(imagePath)
        image = cv2.resize(image, (300, 200))
        output = image.copy()
    
    # pre-process the image for classification
        image = image.astype("float") / 255.0
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        labelactual = imagePath.split(os.path.sep)[-2]
    # classify the input image
        print("[INFO] classifying image...")
        proba = model.predict(image)[0]
        idx = np.argmax(proba)
        label = lb.classes_[idx]

    # we'll mark our prediction as "correct" of the input image filename
    # contains the predicted label text (obviously this makes the
    # assumption that you have named your testing image files this way)
    # build the label and draw the label on the image
        labels ="Predicted {}: {:.2f}%".format(label, proba[idx] * 100,"")
        labelaktual = "Actual {} ".format(labelactual) 
        output = imutils.resize(output, width=400)
        if(label == labelactual):
            correct = "Correct"
            cv2.putText(output, correct, (10, 75),  cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 255, 0), 2)
            cv2.putText(output, labels, (10, 50),  cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 255, 0), 2)
        else:
            correct = "Incorrect"
            cv2.putText(output, correct, (10, 75),  cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 0, 255), 2)
            cv2.putText(output, labels, (10, 50),  cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 0, 255), 2)
        cv2.putText(output, labelaktual, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 255, 0), 2)
        
        # show the output image
        cv2.imwrite('Images/Classify/classify'+str(no)+'.png',output)
        x_label = np.array(lb.classes_)

        y_label = np.arange(len(lb.classes_))
    #  print(df)
        bawal = mpatches.Patch(color='red', label='Bawal {:.2f}%'.format(proba[0] * 100))
        belut = mpatches.Patch(color='blue', label='Belut {:.2f}%'.format(proba[1] * 100))
        lele = mpatches.Patch(color='grey', label='Lele {:.2f}%'.format(proba[2] * 100))
        masorange = mpatches.Patch(color='yellow', label='Mas Orange {:.2f}%'.format(proba[3] * 100))
        masputih = mpatches.Patch(color='aqua', label='Mas Putih {:.2f}%'.format(proba[4] * 100))
        mujaer = mpatches.Patch(color='green', label='Mujaer {:.2f}%'.format(proba[5] * 100))
        nila = mpatches.Patch(color='brown', label='Nila {:.2f}%'.format(proba[6] * 100))
        patin = mpatches.Patch(color='orange', label='Patin {:.2f}%'.format(proba[7] * 100))

        plt.figure(figsize=(5,2.66))
        plt.bar(y_label, proba*100,align="center",color=['red','blue','grey','yellow','aqua','green','brown','orange'])
        plt.xticks(y_label, lb.classes_)
        plt.xlabel("Kelas")
        plt.legend(handles=[bawal,belut,lele,masorange,masputih,mujaer,nila,patin])
        plt.savefig('Images/Plot/plot'+str(no)+'.png')
        no = no + 1
    print("[Info] : Finish Classify, Loading result")
    start()

number = 0
def next():
    global number,load,renderplot,renderimage
    if(number<size_datauji-1):
        number = number + 1
        load = Image.open("Images/Plot/plot"+str(number)+".png")
        renderplot = ImageTk.PhotoImage(load)
        imgplot.configure(image=renderplot)
        load = Image.open("Images/Classify/classify"+str(number)+".png")
        renderimage = ImageTk.PhotoImage(load)
        imgikan.configure(image=renderimage)
    else:
        number = 0
        number = number + 1
        load = Image.open("Images/Plot/plot"+str(number)+".png")
        renderplot = ImageTk.PhotoImage(load)
        imgplot.configure(image=renderplot)
        load = Image.open("Images/Classify/classify"+str(number)+".png")
        renderimage = ImageTk.PhotoImage(load)
        imgikan.configure(image=renderimage)

def previous():
    global number,load,renderplot,renderimage
    if(number>1):
        number = number - 1
        load = Image.open("Images/Plot/plot"+str(number)+".png")
        renderplot = ImageTk.PhotoImage(load)
        imgplot.configure(image=renderplot)
        load = Image.open("Images/Classify/classify"+str(number)+".png")
        renderimage = ImageTk.PhotoImage(load)
        imgikan.configure(image=renderimage)
    else:
        number = size_datauji
        number = number - 1
        load = Image.open("Images/Plot/plot"+str(number)+".png")
        renderplot = ImageTk.PhotoImage(load)
        imgplot.configure(image=renderplot)
        load = Image.open("Images/Classify/classify"+str(number)+".png")
        renderimage = ImageTk.PhotoImage(load)
        imgikan.configure(image=renderimage)

def start():
    global number,load,renderplot,renderimage
    load = Image.open("Images/Plot/plot0.png")
    renderplot = ImageTk.PhotoImage(load)
    imgplot.configure(image=renderplot)
    load = Image.open("Images/Classify/classify0.png")
    renderimage = ImageTk.PhotoImage(load)
    imgikan.configure(image=renderimage)
    button_next.configure(state='normal')
    button_previous.configure(state='normal')

root = Tk()
frame_input_model = Frame(root)
frame_input_model.pack()

frame_input_datauji = Frame(root)
frame_input_datauji.pack()

frame_input_label = Frame(root)
frame_input_label.pack()

button_proses_frame = Frame(root)
button_proses_frame.pack()

image_frame = Frame(root)
image_frame.pack()

button_navigation = Frame(root)
button_navigation.pack()

label_judul = Label(frame_input_model,text="Klasifikasi Ikan Air Tawar CNN",font=('Arial',18))
label_judul.pack(fill=X)

label_model = Label(frame_input_model,text="Nama Model")
label_model.pack(side=LEFT,pady=10,padx=10)

input_model = Entry(frame_input_model,width=40)
input_model.pack(side=LEFT,pady=10,padx=10)
 
label_data_uji = Label(frame_input_datauji,text="Folder Data Uji")
label_data_uji.pack(side=LEFT,pady=10,padx=10)

input_data_uji = Entry(frame_input_datauji,width=40)
input_data_uji.pack(side=LEFT,pady=10,padx=10)

label_data_uji = Label(frame_input_label,text="Label Data Uji")
label_data_uji.pack(side=LEFT,pady=10,padx=10)

input_label = Entry(frame_input_label,width=40)
input_label.pack(side=LEFT,pady=10,padx=10)

button_proses = Button(button_proses_frame,text="Proses",width=50,command=proses_aja)
button_proses.pack(side=BOTTOM,fill=X)


button_previous = Button(button_navigation,text="Sebelumnya",state='disabled',width=50,command=previous)
button_previous.pack(side=LEFT,fill=X,pady=10,padx=10)
button_next = Button(button_navigation,text="Selanjutnya",state='disabled',width=50,command=next)
button_next.pack(side=LEFT,fill=X,pady=10,padx=10)

load = Image.open("Images/bar.png")
renderplot = ImageTk.PhotoImage(load)

load = Image.open("Images/fish.png")
renderimage = ImageTk.PhotoImage(load)

        # labels can be text or images
imgplot = Label(image_frame, image=renderplot,text="Grafik")
imgplot.pack(side=LEFT,pady=10,padx=10,fill=X)
# label_persentase = Label(image_frame,text="Grafik")
# label_persentase.pack(side=TOP,fill=X)

imgikan = Label(image_frame, image=renderimage)
imgikan.pack(side=RIGHT,pady=10,padx=10,fill=X)
# label_hasil = Label(image_frame,text="Hasil Klasifikasi")
# label_hasil.pack(side=TOP,fill=X) 

root.mainloop()