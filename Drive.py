import os
os.environ['TF_CP_MIN_LOG_LEVEL'] = '3'
import time
import numpy as np
import base64

from io import BytesIO
import json

from datetime import datetime
import shutil

import socketio
import eventlet 
import eventlet.wsgi
# from PIL import Image
from flask import Flask
# import cv2

# from Model import Model
# from utils import process_image

import threading
from threading import Thread
import time
import multiprocessing
#	create server
sio = socketio.Server()
app = Flask(__name__)

#	Create a model
model = None

START = None

MAX_SPEED = 25
MIN_SPEED = 15

speed_limit = MAX_SPEED

@sio.on('telemetry')
def telemetry(sid, data):
	if data:
		# image = Image.open(BytesIO(base64.b64decode(data['image'])))
		
		try:
			print("Data this time:- ", data)
			# steering = float(data['steering_angle'])
			throttle = float(data['throttle'])
			speed = float(data['speed'])
			# image = np.asarray(image)
			# image = process_image(image)
			# image = np.expand_dims(image, axis=0)
			steering_angle = np.random.uniform(-0.2, 0.2)

			global speed_limit
			if speed > speed_limit:
				speed_limit = MIN_SPEED
			else:
				speed_limit = MAX_SPEED

			throttle = 1.0 - steering_angle**2 - (speed/speed_limit)**2

			print("{:.4f} || {:.4f} || {:.4f}".format(steering_angle, throttle, speed))
			send_control(steering_angle, throttle)
		
		except Exception as e:
			print(e)
	else:
		# print("Printing Manual Control:- ", data)
		print("Emimting manual data")
		START = time.time()

		sio.emit('manual', data={}, skip_sid=True)
		print("Data Emitted")


@sio.on("callback1")
def callback(data):
	print("\n\nCallback received:- ", data)
	
	result = None
	ev = threading.Event()
	avg_time = []

	START = time.time()

	print("Start emitting")
	
	for i in range(10):
		start = time.time()
		result = sio.call("pythonEvent", 
				data="Sandesha aaya hain.")
		
		result = json.loads(result)

		avg_time.append(time.time() - start)


	print("Ended emitting")

	TOT_TIME = time.time() - START

	print("Total time for 100 emits:- ", TOT_TIME)
	print("Avg time:- ", np.mean(avg_time))
	
	print("Data:- ", result['msg'], " Lenght:- ", len(result['array']))

SID = ""

# @sio.on('connect')
# def connect(sid, environ):
# 	global SID
# 	SID = sid
# 	print("\n\nconnect ", sid, " ENV:- ", environ)
    # send_control(0, 0)

@sio.event
def connect(sid, env):
	print("Connected:- ", sid)
	
def send_control(steering_angle, throttle):
	sio.emit(
			"steer",
			data={
				'steering_angle': steering_angle.__str__(),
				'throttle': throttle.__str__()
			}, skip_sid=True
		)

APP = None
def startServer():
	print("Server Started")
	eventlet.wsgi.server(eventlet.listen(('127.0.0.1', 4567)), APP)
	print("Server closed")


if __name__ == "__main__":
	print("Playing game")
	# global APP
	# APP = socketio.Middleware(sio, app)
	app = socketio.Middleware(sio, app)

	print("App created")
	

	eventlet.wsgi.server(eventlet.listen(('127.0.0.1', 4567)), app)
	# thread_server = Thread(target=startServer)
	# thread_server.daemon = True
	# thread_server.start()
	# serverProcess = multiprocessing.Process(target=startServer)
	# serverProcess.start()

	
	print("\n\nStart the exe")
	# time.sleep(10)
	# serverProcess.terminate()
	# time.sleep(10)
	# global  SID
	
	# if thread_server.is_alive():
	# 	print("Server thread will end too")
	# 	thread_server.join()
	# 	print("Server thread ended too")
