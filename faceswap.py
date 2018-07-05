# main script
import sys

import lib.cli as cli

if sys.version_info[0] < 3:
	raise Exception("This Program requires atleast Python3.2")
if sys.version_info[0] == 3 and sys.version_info[1] < 2:
	raise Exception("This Program requires atleast Python3.2")

def bad_args(args):
	parser.print_help()
	exit(0)

from lib.utils import FullHelpArgumentParser
from script.extract import ExtractTrainingData
from script.train import TrainingProcessor
#from script.convert import ConvertImage

if __name__ == '__main__':
	#parser
	parser = FullHelpArgumentParser()
	subparser = parser.add_subparsers()
	#to extract the face from the images
	extract = ExtractTrainingData(subparser,"extract",
		"Extract the faces from the picture")
	#to train the whole autoencoder network
	train = TrainingProcessor(subparser,"train",
		"To Train a model for two faces  A and B")
	#convert src image to new one with face swapped
	"""convert = ConvertImage(subparser,"convert",
		"Convert a src image  into Target Image")"""


	parser.set_defaults(func=bad_args)
	#parse the arguments
	arguments = parser.parse_args()
	arguments.func(arguments)