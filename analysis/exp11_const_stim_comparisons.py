"""
See writeup. https://docs.google.com/document/d/14KvRBBwdQCg6zsKXNWArELXbpAEcRHKj2LAZfwJN_ns/edit#heading=h.2a81n2ujrig6
Apply various constant stimulus inputs across different model setups and dump membrane traces.
"""

import numpy as np
import pylab as plt
import project_path
from model.data_accessor import get_data_file_abs_path
from model.neuron_metadata import *
from model.neural_model import NeuralModel
import model.init_conds as init_conds
from util.plot_util import *

import pdb

neurons_to_observe = ["PLML", "PLMR"]

neuron_metadata_collection = NeuronMetadataCollection.load_from_chem_json(get_data_file_abs_path('chem.json'))
N = neuron_metadata_collection.get_size()
model = NeuralModel(neuron_metadata_collection)

def main():
  for neurons_to_stimulate in [["PLML"], ["PLML","PLMR"]]:
    for stim_amp_nA in [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 2, 5]:
      case_suffix = "_".join(neurons_to_stimulate) + "_" + str(stim_amp_nA)
      run_one_case(
        case_name = "cook_static_vth_" + case_suffix,
        neurons_to_stimulate = neurons_to_stimulate,
        stim_amp_nA = stim_amp_nA,
        use_cook_connectome = True,
        use_static_vth = True
        )
      run_one_case(
        case_name = "varshney_static_vth_" + case_suffix,
        neurons_to_stimulate = neurons_to_stimulate,
        stim_amp_nA = stim_amp_nA,
        use_cook_connectome = False,
        use_static_vth = True
        )
      run_one_case(
        case_name = "cook_moving_vth_" + case_suffix,
        neurons_to_stimulate = neurons_to_stimulate,
        stim_amp_nA = stim_amp_nA,
        use_cook_connectome = True,
        use_static_vth = False,
        )
      run_one_case(
        case_name = "varshney_moving_vth_" + case_suffix,
        neurons_to_stimulate = neurons_to_stimulate,
        stim_amp_nA = stim_amp_nA,
        use_cook_connectome = False,
        use_static_vth = False,
        )

def run_one_case(case_name, neurons_to_stimulate, stim_amp_nA, use_cook_connectome, use_static_vth):
  """
  Run the model once and save dynamics dump
  -----
  Parameters
  -----
  case_name (str) - Name of the case. Will be used for debugging, and for filename + .png
  neurons_to_stimulate (list(str)) - Neurons to inject constant current to.
  stim_amp_nA (double) - The stimulus magnitude to be applied to all 'neurons_to_stimulate'
  use_cook_connectome (boolean) - If true, use Cook's connectome. Else, use Varshney's
  use_static_vth (boolean) - If true, use static vth, else use the default moving Vth.
  """
  stimulus = {}
  for neuron in neurons_to_stimulate:
    stimulus[neuron] = stim_amp_nA

  # How many timesteps to run simulation for.
  simul_time = 500

  init_conds = 10**(-4)*np.random.normal(0, 0.94, 2*N)


  model = NeuralModel(neuron_metadata_collection)
  model.init_conds = init_conds

  model.set_I_ext_constant_currents(stimulus)

  if use_cook_connectome:
    model.init_kunert_2017_cook_connectome()
  else:
    model.init_kunert_2017()

  model.keep_vth_static = use_static_vth

  (v_mat, s_mat, v_normalized_mat) = model.run(simul_time)

  truncated_potentials = v_mat[200:,:]

  fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(5, 5))
  plot_potentials(
    neuron_names_to_show = neurons_to_observe,
    dynamics = v_mat,
    is_normalized_v = False,
    dt = model.dt,
    neuron_metadata_collection = neuron_metadata_collection,
    fig_axes = [axes[0], axes[1]])

  fig_title = case_name
  fig.suptitle(fig_title, fontsize=16)
  fig.savefig("../local_results/exp11_const_stim_comparisons/" + fig_title + ".png")
  plt.close(fig)

main()