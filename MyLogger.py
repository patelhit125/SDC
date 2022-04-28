import os
import logging
logging.basicConfig(level=logging.INFO)
logging.root.removeHandler(logging.root.handlers[0])

class Logger:
	logDir = "./Logs"
	@staticmethod
	def init(path=None):
		if path is None:
			path = Logger.logDir

		else:
			Logger.logDir = path

		if not os.path.isdir(path):
			os.mkdir(path)

	@staticmethod
	def printLog():
		pass

	@staticmethod
	def info(data, *args):
		if len(args) == 0:
			print(f"\n<------ {data} ------>\n")
		else:
			print(f"\n<------ {data} : {args} ------>\n")
	@staticmethod
	def title(data):
		print(f"\n<======| {data} |======>\n".title())

	@staticmethod
	def createCustomLogger(module=None, filename=None, mode='a', loglevel=logging.WARNING):
		if module is None:
			raise ValueError("Must provide a moudle name.")

		logger = logging.getLogger(module)
		formatter = logging.Formatter('(%(name)-9s) - [%(levelname)s] - %(message)s')

		if filename is not None:
			filepath = os.path.join(Logger.logDir, filename)
			# if os.path.isfile(filepath):

			fileHandler = logging.FileHandler(filepath, mode=mode)
			fileHandler.setLevel(loglevel)
			fileHandler.setFormatter(formatter)
			logger.addHandler(fileHandler)

		else:
			streamHandler = logging.StreamHandler()
			streamHandler.setFormatter(formatter)
			logger.addHandler(streamHandler)

		return logger
	