import os
import argparse

from simwatchdog.io import input_read_func
from simwatchdog.testbed.run_func import run


# Set up argument parsing
parser = argparse.ArgumentParser(description='Run a silent failure benchmark: surrogate, detectors and scoring against the FOM.')
parser.add_argument('input_file', type=str, help='Path to the TOML input file.')
parser.add_argument('--working_directory', type=str, default=None, help='Where results are saved (default: folder of the input file).')

args = parser.parse_args()

# Check if the input file exists
if not os.path.isfile(args.input_file):
    print(f"Error: The input file '{args.input_file}' does not exist.")
    exit(1)

working_dir = args.working_directory or os.path.dirname(os.path.abspath(args.input_file))

# read the input file and fill the defaults
input_param = input_read_func.read_input_file(args.input_file)
config      = input_read_func.init_config(input_param)

results = run(config, working_dir)
