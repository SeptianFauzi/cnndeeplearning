import matplotlib
matplotlib.use("Agg")

# import the necessary packages
from tensorflow.python.keras.callbacks import TensorBoard
from keras.callbacks im	port CSVLogger
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelBinarizer
from septiannet import Septiannet
import matplotlib.pyplot as plt
from imutils import paths
import numpy as np
import argparse
from time import time
import random as rn
import pickle
import cv2
import tensorflow as tf
import os
from tkinter import *
from tkinter.ttk import *
def proses_aja():

	os.environ['PYTHONHASHSEED'] = '0'
	np.random.seed(2)
	rn.seed(2)
	tf.set_random_seed(2)

	from keras import backend as K
	session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,inter_op_parallelism_threads=1)
	sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
	K.set_session(sess)

	ap = argparse.ArgumentParser()
	ap.add_argument("-p", "--plot", type=str, default="plot.png",
		help="path to output accuracy/loss plot")
	args = vars(ap.parse_args())

	EPOCHS = 300 # variabel epoch
	BS = 32 # batchsize
	IMAGE_DIMS = (300, 200, 3)
	tensorboard = TensorBoard(log_dir='logs/'+input_model.get(), write_graph=True, write_images=True, histogram_freq=0)
	csv_logger = CSVLogger(input_model.get()+'.csv')
	early_stopping = EarlyStopping(verbose=1)
	modelcheckpoint = ModelCheckpoint('{epoch:02d}-{val_loss:.2f}-000055.hdf5',verbose=1,save_best_only=False,period=50)
	

	data = []
	labels = []

	datatest = []
	labeltest = [] 

	print("[INFO] loading images...")
	imagePaths = sorted(list(paths.list_images(input_data_latih.get())))

	for imagePath in imagePaths:

		image = cv2.imread(imagePath)
		image = cv2.resize(image, (IMAGE_DIMS[0], IMAGE_DIMS[1]))
		image = img_to_array(image)
		data.append(image)
	

		label = imagePath.split(os.path.sep)[-2]
		labels.append(label)


	data = np.array(data, dtype="float") / 255.0
	labels = np.array(labels)
	print("[INFO] data matrix: {:.2f}MB".format(
		data.nbytes / (1024 * 1000.0)))
	

	lb = LabelBinarizer()
	labels = lb.fit_transform(labels)
	
	imagePathstest = sorted(list(paths.list_images(input_data_uji.get())))

	for imagePath in imagePathstest:

		image = cv2.imread(imagePath)
		image = cv2.resize(image, (IMAGE_DIMS[0], IMAGE_DIMS[1]))
		image = img_to_array(image)
		datatest.append(image)
	

		label = imagePath.split(os.path.sep)[-2]
		labeltest.append(label)


	datatest = np.array(datatest, dtype="float") / 255.0
	labeltest = np.array(labeltest)
	print("[INFO] data matrix: {:.2f}MB".format(
		datatest.nbytes / (1024 * 1000.0)))
	

	lbs = LabelBinarizer()
	labeltest = lb.fit_transform(labeltest)
	

	aug = ImageDataGenerator()


	print("[INFO] compiling model...")
	model = Septiannet.build(width=IMAGE_DIMS[0], height=IMAGE_DIMS[1],
		depth=IMAGE_DIMS[2], classes=len(lb.classes_))
	opt = SGD(lr=0.00055) # variabel learning rate
	model.compile(loss="categorical_crossentropy", optimizer=opt,
		metrics=["accuracy"])
	model.summary()
	
	# train the network
	print("[INFO] training network...")
	H = model.fit(x=data,y=labels,batch_size=BS,epochs=EPOCHS, shuffle=False, verbose=1, callbacks=[tensorboard, csv_logger,modelcheckpoint], validation_data=(datatest,labeltest))
	#save the model to disk
	print("[INFO] serializing network...")
	model.save(input_model.get())
	
	# save the label binarizer to disk
	print("[INFO] serializing label binarizer...")
	f = open(input_label.get(), "wb")
	f.write(pickle.dumps(lb))
	f.close()

	#plot the training loss and accuracy
	plt.style.use("ggplot")
	plt.figure()
	plt.plot(H.history["acc"])
	plt.plot(H.history["val_acc"])
	plt.title("Akurasi Model")
	plt.xlabel("Epoch")
	plt.ylabel("Akurasi ")
	plt.legend(['data latih','data uji'])
	plt.savefig("Model Accuracy" + args["plot"])
	#summarize
	plt.style.use("ggplot")
	plt.figure()
	plt.plot(H.history["loss"])
	plt.plot(H.history["val_loss"])
	plt.title("Loss Model")
	plt.xlabel("Epoch")
	plt.ylabel("Loss")
	plt.legend(['data latih','data uji'])
	plt.savefig("Model Loss" + args["plot"])

root = Tk()
frame_input_model = Frame(root)
frame_input_model.pack()

frame_input_datalatih = Frame(root)
frame_input_datalatih.pack()

frame_input_datauji = Frame(root)
frame_input_datauji.pack()


frame_input_label = Frame(root)
frame_input_label.pack()

button_proses_frame = Frame(root)
button_proses_frame.pack()


label_judul = Label(frame_input_model,text="Training Klasifikasi Ikan Air Tawar CNN",font=('Arial',18))
label_judul.pack(fill=X)

label_model = Label(frame_input_model,text="Nama Model")
label_model.pack(side=LEFT,pady=10,padx=5)

input_model = Entry(frame_input_model,width=40)
input_model.pack(side=LEFT,pady=10)

label_data_latih = Label(frame_input_datalatih,text="Folder Data Latih")
label_data_latih.pack(side=LEFT,pady=10,padx=5)

input_data_latih = Entry(frame_input_datalatih,width=40)
input_data_latih.pack(side=LEFT,pady=10)

label_data_uji = Label(frame_input_datauji,text="Folder Data Uji")
label_data_uji.pack(side=LEFT,pady=10,padx=5)

input_data_uji = Entry(frame_input_datauji,width=40)
input_data_uji.pack(side=LEFT,pady=10)

label_data_uji = Label(frame_input_label,text="Label")
label_data_uji.pack(side=LEFT,pady=10,padx=5)

input_label = Entry(frame_input_label,width=40)
input_label.pack(side=LEFT,pady=10)

button_proses = Button(button_proses_frame,text="Proses",width=50,command=proses_aja)
button_proses.pack(side=BOTTOM,fill=X)

root.mainloop()