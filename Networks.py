import os
import tensorflow as tf
from tensorflow.keras.layers import Dense, Lambda, Flatten
from tensorflow.keras import Model, Sequential

class CriticNetwork(Model):
	"""Critic Network for providing a state-action value."""
	def __init__(self, fc_dims):
		"""
		Arguments:
			fc_dim: List of Integer: specifying the dimensions for ff layers
		"""

		super(CriticNetwork, self).__init__()
		self.fc_dims = fc_dims


		self.thisLayers = []
		for dim in self.fc_dims:
			self.thisLayers.append(Dense(dim, activation='relu'))
			
		self.q = Dense(1, activation=None)

	def call(self, state, action):
		prob = self.thisLayers[0](tf.concat([state, action], axis=1))
		
		for layer in self.thisLayers[1:]:
			prob = layer(prob)

		q = self.q(prob)

		return q

class ValueNetwork(Model):
	"""Value network to provide for the state value"""
	def __init__(self, fc_dims):
		"""
		Arguments:
			fc_dim: List of Integeger: specifying the dimensions for Dense layers
		"""
		super(ValueNetwork, self).__init__()
		self.fc_dims = fc_dims

		self.thisLayers = []
		for dim in self.fc_dims:
			self.thisLayers.append(Dense(dim, activation='relu'))

		self.v = Dense(1, activation=None)

	def call(self, state):
		
		state_value = self.thisLayers[0](state)
		
		for layer in self.thisLayers[1:]:
			state_value = layer(state_value)

		v = self.v(state_value)
		return v

class ActorNetwork(Model):
	"""Actor network our main agent that will learn to play in environment"""
	def __init__(self, fc_dims, n_action):
		"""
		Arguments:
			fc_dim: List of Integeger: specifying the dimensions for Dense layers
			n_action: Integer specifying the number of action a model can take
		"""
		super(ActorNetwork, self).__init__()
		self.fc_dims = fc_dims
		self.n_action = n_action

		self.thisLayers = []
		for dims in self.fc_dims:
			self.thisLayers.append(Dense(dims, activation='relu'))

		self.mu = Dense(self.n_action, activation='linear')
		self.sigma = Dense(self.n_action, activation='linear')

	def call(self, state):
		prob = self.thisLayers[0](state)
		
		for layer in self.thisLayers[1:]:
			prob = layer(prob)

		mu = self.mu(prob)
		sigma = self.sigma(prob)

		sigma = tf.clip_by_value(sigma, -10, 2)
		sigma = tf.exp(sigma)

		return mu, sigma

	# Sigma value clipped from -20,2 to -10,2