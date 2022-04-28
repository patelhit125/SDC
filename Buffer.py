import numpy as np
import os
import json

class ReplayBuffer(object):
	"""Replay Buffer to store transition for training the SAC network from past experiences"""
	def __init__(self, max_exp_size, history_len, input_dim, n_action):
		"""	
		Arguments:
			max_exp_size: Integer, Maximum number of experiences to store
			history_len: Intenger, Number of obs to stack to create a stack for agent
			input_dim: Tupple, Define the shape of the observation
			n_action: Integer, Stating the number of action an agent can take
		"""
		self.max_exp_size = max_exp_size
		
		self.input_dim = input_dim
		self.n_action = n_action
		self.history_len = history_len

		self.state = np.empty((self.max_exp_size, *self.input_dim), dtype=np.float32)
		self.action = np.empty((self.max_exp_size, self.n_action), dtype=np.float32)
		self.reward = np.empty(self.max_exp_size, dtype=np.float32)
		self.done = np.empty(self.max_exp_size, dtype='bool')

		self.count = 0
		self.current = 0

	def store_transition(self, s, a, r, d):
		"""Store an experience for a timestamp
		Arguments:
			s: An array of observation obtained from environment
			a: An array of size (n_action,) where each value is in range (-1, 1)
			r: A float specifying the perfomance for taking an action
			d: A bool indicating the termination of an episode

		"""
		self.state[self.current] = s
		self.action[self.current] = a
		self.reward[self.current] = r
		self.done[self.current] = d

		self.current = (self.current+1) % self.max_exp_size
		self.count = max(self.count, self.current)

	def getBatchData(self, batch_size):
		"""Returns a batch for training the agent
		Arguments:
			batch_size: Integer, specifying the size of batch to train Agent on
		Returns:
			A tupple containing state, action, reward, done, next_state
		"""
		
		batch = np.random.choice(self.count, size=batch_size)
		s = self.state[batch]
		a = self.action[batch]
		r = self.reward[batch]
		d = self.done[batch]
		s_ = self.state[batch+1]
		
		return s, a, r, d, s_

	def save(self, dir_name):
		"""Save the buffer to specified folder
		Argument:
			dir_name: String, path to the buffer directory for nth checkpoint
		"""
		np.save(dir_name + "/state.npy", self.state[:self.count])
		np.save(dir_name + "/action.npy", self.action[:self.count])
		np.save(dir_name + "/reward.npy", self.reward[:self.count])
		np.save(dir_name + "/done.npy", self.done[:self.count])

		varDir = {
			'count': self.count,
			'current': self.current
		}

		with open(os.path.join(dir_name, 'vars.json'), 'w') as jfile:
			json.dump(varDir, jfile)


	def load(self, dir_name):
		"""Load the buffer from a specified folder
		
		Argument:
			dir_name: String, path to the buffer directory for nth checkpoint
		"""
		super()

		f = open(os.path.join(dir_name, 'vars.json'), 'r')
		data = json.load(f)
		self.count = data['count']
		self.current = data['current']

		self.state[:self.count] = np.load(dir_name + "/state.npy")
		self.action[:self.count] = np.load(dir_name + "/action.npy")
		self.reward[:self.count] = np.load(dir_name + "/reward.npy")
		self.done[:self.count] = np.load(dir_name + "/done.npy")

		print("<---- Buffer Loaded ---->")