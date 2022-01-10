import sys, getopt
from helper_functions import *
from structures.learner import Learner
from train_system import TrainSystem


def get_args(opts):

	args = {
		"padding": "SAME"
	}

	for opt, arg in opts:
		if opt == "--batch_size":
			args["batch_size"] = int(arg)
		elif opt == "--epochs":
			args["epochs"] = int(arg)
		elif opt == "--learner_epochs":
			args["learner_epochs"] = int(arg)
		elif opt == "--num_classes":
			args["num_classes"] = int(arg)
		elif opt == "--steps_per_epoch_limit":
			args["steps_per_epoch_limit"] = int(arg)
		elif opt == "--num_shot":
			args["num_shot"] = int(arg)
		elif opt == "--num_shot_eval":
			args["num_shot_eval"] = int(arg)
		elif opt == "--use_bias":
			args["use_bias"] = bool(int(arg))
		elif opt == "--image_x":
			args["img_x"] = int(arg)
		elif opt == "--image_y":
			args["img_y"] = int(arg)
		elif opt == "--filters":
			args["filters"] = int(arg)
		elif opt == "--padding":
			args["padding"] = arg.strip().upper()
		elif opt == "--kernel_size":
			args["kernel_size"] = int(arg)
		elif opt == "--num_blocks":
			args["num_blocks"] = int(arg)
		elif opt == "--maxpool_size":
			args["maxpool_size"] = int(arg)
		elif opt == "--meta_hidden_size":
			args["meta_hidden_size"] = int(arg)
		elif opt == "--meta_input_size":
			args["meta_input_size"] = int(arg)
		elif opt == "--b_init_0":
			args["bF_init_0"] = float(arg)
			args["bI_init_0"] = -float(arg)
		elif opt == "--b_init_1":
			args["bF_init_1"] = float(arg)
			args["bI_init_1"] = -float(arg)

	return args

def main():
	opts, _ = getopt.getopt(sys.argv[1:], shortopts = "", longopts=["batch_size=", "epochs=", "learner_epochs=", "num_classes=",
		"steps_per_epoch_limit=", "num_shot=", "num_shot_eval=", "use_bias=", "image_x=", "image_y=",
		"filters=", "padding=", "kernel_size=", "num_blocks=", "maxpool_size=", "meta_hidden_size=",
		"meta_input_size=", "b_init_0=", "b_init_1="])
	
	args = get_args(opts)

	dummy_learner = Learner(args)

	args["learner_params"] = calc_learner_params(dummy_learner.model)

	train_system = TrainSystem(args)

	train_system.train()

	checkpoint(train_system.meta_learner, "meta_learner_weights")

if __name__ == "__main__":
	main()





		


		

