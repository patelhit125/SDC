import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from Environment import CarSim
from Agent import Agent
from config import CONFIG

import multiprocess
import threading

import time
import traceback
import socketio
import eventlet
import eventlet.wsgi 
eventlet.monkey_patch()

from flask import Flask

event = threading.Event()
SID = None

sio = socketio.Server()
app = Flask(__name__)
app = socketio.Middleware(sio, app)

@sio.on('connect')
def connect(sid, env):
	global SID
	print("Connected to client: ", sid)
	SID = sid
	event.set()

def launchServer():
	global app
	print("Server has Started. Launch Exe in 60s")
	eventlet.wsgi.server(eventlet.listen(('127.0.0.1', 3000)), app)

def recordVideo():

	env = CarSim(sio)
	agent = Agent(env.observation_space.shape, env.action_space.shape[0], batch_size=CONFIG['BATCH_SIZE'])
	agent.load()

	obs = env.reset()
	env.createVideoInstance()

	done = False
	reward = 0
	frames = 0
	start = time.time()

	while not done:
		action, _ = agent.get_action(obs)
		new_obs, r, done = env.step(action)

		reward += r
		frames += 1
		env.writeFrame()

		obs = new_obs

	tot_time = time.time() - start

	print("Frames: %3d || Reward: %.2f || Time: %.2f"%(frames, reward, tot_time))
	env.releaseVideoInstance()

if __name__ == "__main__":

	testingThread = None

	try:
		serverProcess = threading.Thread(target=launchServer, name="ServerThread")
		serverProcess.daemon = True
		serverProcess.start()

		print("Process Started")

		if event.wait():
			print("Starting testing")

			testingThread = sio.start_background_task(target=recordVideo)
		else:
			raise Exception("Failed to connect to server...")
	except Exception as e:			
		traceback.print_exc()

	finally:
		print("Waiting for thread to end")

		if testingThread is not None: testingThread.join()
