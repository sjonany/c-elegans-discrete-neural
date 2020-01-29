"""
Script for experiment 1: Constant stimulus, single parameter at a time.
This generates a bunch of images at the local_results/exp1/ directory
This is the automation of the single_param notebook.

Usage:
1. Modify the variable "unparsed_cases" in this file
2. Run this file. "python exp1_const_stim.py"
3. See simulation results for each test case in the {ROOT}/local_results/exp1_const_stim
"""

# Initialize helpers.
import numpy as np
import project_path
from model.data_accessor import get_data_file_abs_path
from model.neuron_metadata import *
neuron_metadata_collection = NeuronMetadataCollection.load_from_chem_json(get_data_file_abs_path('chem.json'))
N = neuron_metadata_collection.get_size()

##############################################################################
# START of variables you might want to change

# All simulation results will be dumped to this directory.
output_dir = "../local_results/exp1_const_stim"

# You should only have to modify this variable.
# Each string is its own case, where you specify the neurons and stimulus amplitude.
# If a neuron name is just three letters long, we will expand it with L and R.
unparsed_cases = [
  "AIY:1",
  "AIY:5",
  "ALM:1",
  "ALM:5",
  "ASE:1",
  "ASE:5",
  "ASH: 1",
  "ASH: 5",
  "ASK: 1",
  "ASK: 5",
  "AVA:1",
  "AVA:5",
  "AVD:1",
  "AVD:5",
  "AVE:1",
  "AVE:5",
  "AWA:1",
  "AWA:5",
  "AWB:1",
  "AWB:5",
  "AWC:1",
  "AWC:5",
  "IL1:1",
  "IL1:5",
  "IL2:1",
  "IL2:5",
  "PLM:1.4",
  "PLM:5",
  "RIV:1",
  "RIV:5",
  "RMD:1",
  "RMD:5",
  "PLML: 1, PLMR: 5",
  "ASKL: 1, ASKR: 5",
  "ASHR: 1",
  "ASHR: 5",
  "ASHL: 1, ASHR: 5",
  "PLM: 1.4, AVB: 2.3",
  "PLM: 1.4, AVB: 2.3, ALM: 1, AVM: 1",
  "PLM: 1.4, AVB: 2.3, ALM: 5, AVM: 5",
  "PLM: 1.4, AVB: 2.3, RIV: 1",
  "PLM: 1.4, AVB: 2.3, RIV: 5",
  "ASK: 1.4, PLM: 1.4",
  "ASK: 5, PLM: 1.4"
]

# Neurons to plot membrane potentials for.
neurons_to_observe = NeuronMetadataCollection.create_lr_names_from_base([
  "AIY",
  "ALM",
  "ASE",
  "ASH",
  "ASK",
  "AWA",
  "AWB",
  "AWC",
  "ASK",
  "IL1",
  "IL2",
  "PLM",
  "RIV",
  "RMD"
  ])

# How many timesteps to run simulation for.
simul_time = 2000

# How many repetitions per case. You might want multiple when using random initial conditions.
num_iter = 3

# END of variables you might want to change
####################################################################################

import matplotlib.pyplot as plt
import os
import sys
import time
from model.neural_model import NeuralModel
from sklearn.decomposition import PCA
from util.plot_util import *

# Array of cases
# Case = dictionary of full neuron name (string) to constant current strength (double)
# E.g. [{'AIYL': 1.0}, {'IL2L': 5.0}]
cases = []
for unparsed_case in unparsed_cases:
  neuron_to_amp = {}
  neuron_amp_strs = unparsed_case.split(",")
  for neuron_amp_str in neuron_amp_strs:
    neuron_amp = neuron_amp_str.split(":")
    neuron = neuron_amp[0].strip().upper()
    amp = float(neuron_amp[1].strip())

    # Expand base neuron names like "AIY" to two inputs: "AIYL", "AIYR"
    # AVM is an exception. There is no L/R.
    if neuron.endswith("L") or neuron.endswith("R") or neuron == "AVM":
      neuron_to_amp[neuron] = amp
    else:
      neuron_to_amp[neuron + "L"] = amp
      neuron_to_amp[neuron + "R"] = amp
  cases.append(neuron_to_amp)

if os.path.isdir(output_dir):
  print("Output directory {0} already exists." + \
    " Note that we do NOT overwrite pre-existing cases.".format(output_dir))
else:
  os.mkdir(output_dir)

# Helper method to produce one simulation.
def gen_images_one_simulation(case_subdir_path, file_prefix):
  start_time = time.time()
  # Run model
  model = NeuralModel(neuron_metadata_collection)

  # Initial condition
  # If you want a fixed-seed initial condition, uncomment the line below.
  # np.random.seed(0)
  model.init_conds = 10**(-4)*np.random.normal(0, 0.94, 2*N)

  model.set_I_ext_constant_currents(case)
  model.init()
  (v_mat, s_mat, v_normalized_mat) = model.run(simul_time)

  # The oscillatory dynamic doesn't stabilize until about dt*300 onwards.
  # Also, interactome analysis is done after the first 50 timesteps.
  truncated_potentials = v_normalized_mat[300:,:]

  fig = plot_potentials(neurons_to_observe, truncated_potentials, neuron_metadata_collection)
  fig.savefig(os.path.join(case_subdir_path, "{0}neurons.png".format(file_prefix)), dpi=36)
  plt.close(fig)

  for neuron_type in [NeuronType.SENSORY, NeuronType.MOTOR, NeuronType.INTERNEURON]:
    relevant_ids = neuron_metadata_collection.get_neuron_ids_by_neuron_type(neuron_type)
    relevant_potentials = truncated_potentials[:, relevant_ids]
    num_pca_comps = 4
    pca = PCA(n_components = num_pca_comps)
    projected_X = pca.fit_transform(relevant_potentials)

    dt = 0.01
    times = np.arange(0, projected_X.shape[0] * model.dt , model.dt)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 3))
    for pca_comp in range(num_pca_comps):
      ax.set_title(neuron_type)
      ax.plot(times, projected_X[:,pca_comp], label = "PCA mode {0:d}".format(pca_comp+1))
      ax.legend()
    fig.savefig(os.path.join(case_subdir_path, "{0}pca_{1}.png".format( \
      file_prefix, str(neuron_type))), dpi=36)
    plt.close(fig)
  print("Simulation took %s seconds" % (time.time() - start_time))

for case_i in range(len(cases)):
  print("Working on case {0}/{1}: {2}".format(case_i+1 , len(cases), cases[case_i]))
  case = cases[case_i]
  case_subdir_path = os.path.join(output_dir, str(case))
  if os.path.isdir(case_subdir_path):
    print("{0} already exists. Skipping.".format(case_subdir_path))
    continue
  
  os.mkdir(case_subdir_path)
  for iter_i in range(num_iter):
    file_prefix = "iter%d_" % (iter_i+1)
    gen_images_one_simulation(case_subdir_path, file_prefix)
