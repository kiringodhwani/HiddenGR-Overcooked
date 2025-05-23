import os


# ROOT_DIR = os.environ['RLROOT']
# Trying to make sure that this works even without the environment
# print(os.environ['RLROOT'])
ROOT_DIR = os.environ.get('RLROOT', os.path.abspath(os.path.realpath(__file__)+"../../.."))

DEFAULT_PARAMS = {
    'episodes': 1000000,
    'training_steps': 1000000,
    'batch_size': 32,
    'mem_size': 200000,
    'dry_size': 10000,
    'input_shape': (4, 90),
    'history': 4,
}

GOAL_REWARD = 1.
TIMESTEP_REWARD = 0.
INVALID_ACTION_REWARD = -1.
PARTIAL_GOAL_REWARD = 2.