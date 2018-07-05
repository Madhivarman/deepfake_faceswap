#import necessary libraries
import cv2
import numpy
import time

from threading import Lock
from lib.utils import get_image_paths,get_folder
from lib.cli import FullPaths
from plugins.PluginLoader import PluginLoader

class TrainingProcessor(object):
	arguments  = None

	def __init__(self, subparser, command, description='default'):
		self.parse_arguments(description,subparser,command)
		self.lock = Lock()

	def process_arguments(self,arguments):
		self.arguments = arguments
		print("Model A Directory:{}".format(self.arguments.input_A))
		print("Model B Directory:{}".format(self.arguments.input_B))
		print("Training Data Directory:{}".format(self.arguments.model_dir))


		self.process() #call the function


	def parse_arguments(self,description,subparser,command):
		parser = subparser.add_parser(
			command,
			help="This command trains the model for Two faces A and B directory",
			description = description)

		parser.add_argument('-A','--input-A',
			action=FullPaths,
			dest='input_A',
			default='input_A',
			help="Input Directory. A directory containing training images for Face A")


		parser.add_argument('-B','--input-B',
			action=FullPaths,
			dest="input_B",
			default="input_B",
			help="Input Directory. A directory containing training images for Face B")


		parser.add_argument('-m','--model-dir',
			action=FullPaths,
			dest="model_dir",
			default="models",
			help="Model Directory. This is were Trained Data will stored")

		parser.add_argument('-p','--preview',
			action="store_true",
			dest="preview",
			default="False",
			help="Show Preview Output.")

		parser.add_argument('-v','--verbose',
			action="store_true",
			dest="verbose",
			default=False,
			help="Show Verbose Output")

		parser.add_argument('-s','--save-interval',
			type=int,
			dest="save_interval",
			default=100,
			help="Sets the number of iterations before saving the model")

		parser.add_argument('-w','--write_image',
			action="store_true",
			dest="write_image",
			default=False,
			help="Writes the Training Result to a file even on a preview mode.")


		parser.add_argument('-t','--trainer',
			type=str,
			choices=("Original","LowMem","GAN"),
			default="GAN",
			help="Select Trainer to use, LowMem for cards will take < 2gb.")

		parser.add_argument('-bs','--batch-size',
			type=int,
			default=32,
			help="Batch size, as a power of 2 (64,128,256, etc)")


		parser = self.add_optional_arguments(parser)
		parser.set_defaults(func=self.process_arguments)


	def add_optional_arguments(self,parser):

		return parser


	def process(self):
		#threading
		import threading

		self.stop = False
		self.save_now = False

		thr =  threading.Thread(target=self.processThread, args=(), kwargs={})
		thr.start()

		if self.arguments.preview:
			print("Using Live Preview")
			while True:
				try:
					with self.lock:
						for name,image in self.preview_buffer.items():
							cv2.imshow(name,image)

					#wait for keyboard key
					key = cv2.waitKey(1000)
					if key == ord('\n') or key == ord('\r'):
						break
					if key == ord('s'):
						self.save_now =  True


				except KeyboardInterrupt:
					break

		else:
			input() #how to catch a specific key instead of Enter

		print("Exit Requested..! The trainer will complete its current cycle, save the model and quit")
		self.stop = True
		thr.join() #waits until thread finishes


	def processThread(self):
		print("Loading Data..! This may take a while")

		trainer = self.arguments.trainer
		trainer = "LowMem" if trainer.lower() == "lowmem" else trainer
		model = PluginLoader.get_model(trainer)(get_folder(self.arguments.model_dir))
		model.load(swapped = False)

		images_A = get_image_paths(self.arguments.input_A)
		images_B = get_image_paths(self.arguments.input_B)

		trainer = PluginLoader.get_trainer(trainer)
		trainer = trainer(model,images_A,images_B,batch_size = self.arguments.batch_size)

		try:

			print("Starting. Press Enter to stop Training and Save model")

			for epoch in range(0,100000):
				save_iteration = epoch % self.arguments.save_interval == 0

				trainer.train_one_step(epoch,self.show if (save_iteration or self.save_now) else None)

				if save_iteration:
					model.save_weights()

				if self.stop:
					model.save_weights()
					exit()

				if self.save_now:
					model.save_weights()
					self.save_now = False

		except KeyboardInterrupt:
			try:
				model.save_weights()
			except KeyboardInterrupt:
				print("Saving model weights has been cancelled...!")
			exit(0)


	preview_buffer = {}

	def show(self,image,name=""):
		try:
			if self.arguments.preview:
				with self.lock:
					self.preview_buffer[name] = image

			elif self.arguments.write_image:
				cv2.imwrite('_sample_{}'.format(name),image)

		except Exception as e:
			print("Could not preview Sample..!")
			print(e)
