# General Moudles
import numpy as np
import time
import json
from io import BytesIO
import base64
from PIL import Image
import cv2

from config import CONFIG
import os
#	Environment Modules
import gym
from gym import Env, spaces

class CarSim(Env):
	def __init__(self, sio, history_len=10, video_dir="./videos"):
		super(CarSim, self).__init__()

		self.sio = sio
		self.video_dir = video_dir

		if not os.path.isdir(self.video_dir):
			os.mkdir(self.video_dir)

		self.history_len = history_len

		self.observation_space = spaces.Box(
									low = np.full((self.history_len*10), -1, dtype=np.float32),
									high = np.full((self.history_len*10), 100, dtype=np.float32),
									dtype = np.float32	
								)

		self.action_space = spaces.Box(
									low = np.full((2), -1, dtype=np.float32),
									high = np.full((2), 1, dtype=np.float32),
									dtype=np.float32
								)

		self.observation = np.empty(self.observation_space.shape[0]*self.history_len, dtype=np.float32)
		self.__frame_size = self.__get_frame_size()

		self.video_cnt = 0

	def __get_frame_size(self):
		"""Get size of frame to create VIDEO instance"""
		frame = self.sio.call("render")

		frame = Image.open(BytesIO(base64.b64decode(frame)))
		frame = np.asarray(frame)
		
		return frame.shape


	def __updateObservations(self, new_obs):
		
		self.observation = np.append(self.observation[self.history_len:], new_obs)

	def step(self, action):
		"""
			Step function to perfom action
			parmas: 
				action: a list of array of 2 values
			returns:
				observation: a list of observation obtained after taking action
				reward: Reward obtained for taking action
				done: If episode ended due to action
				info: Additional info
		"""

		result = self.sio.call("step", data={'acceleration':action[0].__str__(), 'steering_angle':action[1].__str__()})
		result = json.loads(result)

		obs = result['observation'] # 10
		self.__updateObservations(obs)

		return self.observation, result['reward'], result['done']
	
	def reset(self):
		"""
			Reset environment after the episode has ended or going to start
			new episode.
			retruns:
				observation: Initial observation
		"""

		result = self.sio.call("reset", data={})
		result = json.loads(result)

		self.observation = np.tile(np.array(result['observation']), self.history_len) # 100
		
		return self.observation
	
	def render(self):
		"""Gets an image of what the car is looking at from center camera"""
		image = self.sio.call('render')

		image = Image.open(BytesIO(base64.b64decode(image)))
		image = np.asarray(image)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

		return image

	def createVideoInstance(self, episode=None):
		"""Create a video instance for recording video
		Arguments:
			episode: Integer, number of episode after which a video instance is created.
			if episode is none than default vidoe count will be consider for naming vidoes.
		"""
		if episode is None:
			episode = self.video_cnt
			self.video_cnt += 1

		self.video_inst = cv2.VideoWriter(f'./videos/Video-{episode}.avi', cv2.VideoWriter_fourcc(*'DIVX'), CONFIG['FPS'], self.__frame_size[:2][::-1])

	def writeFrame(self):
		"""Gets a frame from Unity Env and writes it to file"""
		frame = self.render()
		self.video_inst.write(frame)

	def releaseVideoInstance(self):
		"""Releases the video instance"""
		self.video_inst.release()

	def getFrameSize(self):
		"""Return the size of frame"""
		return self.__frame_size