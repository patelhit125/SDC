import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

#	Communication Modules
import socketio
import eventlet
import eventlet.wsgi
eventlet.monkey_patch()
from flask import Flask

# Multithreading Moduels	
import multiprocess
import threading

#	General modules
import time
import logging
import json
import numpy as np

#	Logging Modules
from MyLogger import Logger
import logging
import traceback

#	Local Modules
from Environment import CarSim
from Agent import Agent

from config import CONFIG

Logger.init()

#	Set logging to info and remove custom StreamHandler
logging.basicConfig(level=logging.WARNING)
logging.root.removeHandler(logging.root.handlers[0])

# ##### Setting Logger #####
socketio_logger = Logger.createCustomLogger('socketio', filename='socket_file.log', mode='w')
sio = socketio.Server(logger=socketio_logger)

event = threading.Event()

SID = None

#	Create a flaskApp with name of current file
app = Flask(__name__)
app = socketio.Middleware(sio, app)

@sio.on('connect')
def connect(sid, env):
	global SID
	Logger.info("Connected:- ", sid)
	SID = sid
	event.set()

def launchServer():
	"""A thread method to run server"""
	global app
	Logger.info("Server has Started. Launch Exe in 60s to start working.")
	eventlet.wsgi.server(eventlet.listen(('127.0.0.1', 3000)), app)
	print("Launch server exited")

def main():
	"""This is the main function where all training will occur"""
	env = CarSim(sio)
	
	agent = Agent(env.observation_space.shape, env.action_space.shape[0], batch_size=CONFIG['BATCH_SIZE'])
	print("....Loading the model....")
	agent.load()

	summary_writer = tf.summary.create_file_writer(CONFIG['TB_LOG']+'/AI-Car{}'.format(int(time.time())))
	
	best_reward = env.reward_range[0]
	score_list = []

	frame_number = 0

	for e in range(CONFIG['EPISODES']):

		#	Record Video instance after specified Episodes
		# if (e+1) % CONFIG['RECORD_VIDEO'] == 0:
		# 	recordVideo(agent, env)
		# 	continue

		obs = env.reset()

		episodic_reward = 0
		start_ep = time.time()

		done = False
		frame_cnt = 0

		while not done:
			action, _ = agent.get_action(obs)
			
			new_obs, reward, done= env.step(action)
			
			frame_number += 1
			
			frame_cnt += 1

			episodic_reward += reward

			agent.remember(obs, action, reward, done)
			agent.learn()

			obs = np.copy(new_obs)

		score_list.append(episodic_reward)
		avg_reward = np.mean(score_list[-100:])

		if avg_reward > best_reward:
			best_reward = avg_reward
		
		with summary_writer.as_default():
			tf.summary.scalar("Score", episodic_reward, step=e)
			tf.summary.scalar("Avg Reward", avg_reward, step=e)
			tf.summary.scalar("Best Reward", best_reward, step=e)

		print("%3d || Score: %.2f ||  Avg Score: %.2f || Best Score: %.2f || Frame: %3d || Time = %3d"%\
			 (e+1, score_list[-1], avg_reward, best_reward, frame_cnt, time.time() - start_ep))	

		#	Save all data after specified Episodes
		if (e+1) % CONFIG['SAVE_DATA'] == 0:
			saveThread = threading.Thread(target=saveAllData, args=(agent, e+1))
			saveThread.start()
			# agent.save()


def saveAllData(agent, episode):
	"""Save all data for loading and reusing later
	Arguments:
		agent: An instance of the Agent class that is used in main() method
	"""
	print("\n<====| Saving data |====>\n")
	
	agent.save(episode)

	print("\n<====| Data Saved |====>\n")

if __name__ == "__main__":

	"""Starting point of process. Thread allocation takes place here so do not change any code."""

	trainingThread = None
	try:
		serverProcess = threading.Thread(target=launchServer, name="ServerThread")
		serverProcess.daemon = True
		serverProcess.start()
		print("Process Started");

		#	waiting for server to be acknowledged in 60 seconds
		if event.wait(60):
			Logger.title("Starting the main loop")
			trainingThread = sio.start_background_task(target=main)
		else:
			raise Exception("Failed to connect to server.")

	except Exception as e:
		traceback.print_exc()

	finally:
		print("Waiting for thread to end")

		if trainingThread is not None: trainingThread.join()
