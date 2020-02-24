"""
See writeup. https://docs.google.com/document/d/1St4qOVr0Bv-Kyh0PwRG6zc75VT38Pm8z5JhOql96W8A/edit#
Unlike ...bruteforce.py, this doesn't just do stairway input, but allows users to provide step stimulus with
varying step durations and amplitudes
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

"""
Parameters
"""
key_neuron = "AWAR"
neurons_to_stimulate = [key_neuron]
neurons_to_observe = [key_neuron]

"""
Each case is composed of the following:
case_name(str) - Name of the case. Will be used for debugging, and for filename + .png
init_cond_seed, init_cond_amp_nA - See init_conds.generate_init_cond_with_seed
  This determines the initial condition of the simulation
stim_spec - a list of amplitude(nA)-duration(s) tuples.
  E.g. [(6.7 nA, 1s), (6.8 nA, 2s)] means the stimulus is 3 seconds long, with the first 1s
  being 6.7 nA
"""
def build_case(case_name, init_cond_seed, init_cond_amp_nA, stim_spec):
  return {
    'case_name': case_name,
    'init_cond_seed': init_cond_seed,
    'init_cond_amp_nA': init_cond_amp_nA,
    'stim_spec': stim_spec
  }

def build_case_from_setup(setup_name, setup_stim_spec, step_duration_s, increment_nA):
  """
  See this section for why these specific setups.
  https://docs.google.com/document/d/1St4qOVr0Bv-Kyh0PwRG6zc75VT38Pm8z5JhOql96W8A/edit#heading=h.ud0sn8353b9h
  """
  case_name = "%s_%.2fs_%.2fnA" % (setup_name, step_duration_s, increment_nA)
  init_cond_seed = 11
  init_cond_amp_nA = 6.7
  num_step = 3
  # Use the setup stim spec to get to the sensitive zone, before adding mini steps
  last_amp_in_setup = setup_stim_spec[-1][0]
  stim_spec = setup_stim_spec + \
      [(last_amp_in_setup + (i + 1) * increment_nA, step_duration_s) for i in range(num_step)]
  return build_case(case_name, init_cond_seed, init_cond_amp_nA, stim_spec)

def build_case_from_setup1(step_duration_s, increment_nA):
  return build_case_from_setup("setup1", [(6.7, 2), (6.718, 2)], step_duration_s, increment_nA)

def build_case_from_setup2(step_duration_s, increment_nA):
  return build_case_from_setup("setup2", [(6.7, 4)], step_duration_s, increment_nA)

cases = []
for increment_nA in [0.001, 0.002, 0.004, 0.008, 0.016]:
  for step_duration_s in [0.2, 0.5, 1, 1.5, 2]:
    cases.append(build_case_from_setup1(step_duration_s, increment_nA))
    cases.append(build_case_from_setup2(step_duration_s, increment_nA))

neuron_metadata_collection = NeuronMetadataCollection.load_from_chem_json(get_data_file_abs_path('chem.json'))
N = neuron_metadata_collection.get_size()
model = NeuralModel(neuron_metadata_collection)
neurons_to_stimulate = [key_neuron]

def run_one_case(case_name, init_cond_seed, init_cond_amp_nA, stim_spec):
  step_amplitudes_nA, step_durations_s = zip(*stim_spec)

  # Times when step values change
  step_timechanges_s = np.concatenate(([0], (np.cumsum(step_durations_s)[:-1])))

  model = NeuralModel(neuron_metadata_collection)
  initial_conds = init_conds.generate_init_cond_with_seed(
      model, key_neuron, init_cond_amp_nA, init_cond_seed)
  model.init_conds = initial_conds

  def time_to_I_ext_fun(t):
    # Find last index of timechanges where current time is greater than t
    amp_i = max(np.searchsorted(step_timechanges_s, t) -1 , 0)
    amp = step_amplitudes_nA[amp_i]
    cur_I_ext = np.zeros(N)
    for neuron in neurons_to_stimulate:
      neuron_id = neuron_metadata_collection.get_id_from_name(neuron)
      cur_I_ext[neuron_id] = amp
    return cur_I_ext

  # These are timesteps when I_ext changes
  t_changes_I_ext = step_timechanges_s.tolist()
  simul_timesteps = int(sum(step_durations_s) / model.dt)

  model.set_I_ext(time_to_I_ext_fun, t_changes_I_ext)
  model.init_kunert_2017()

  # Run the model
  (v_mat, s_mat, v_normalized_mat) = model.run(simul_timesteps)

  fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))
  plot_potentials(
    neuron_names_to_show = neurons_to_observe,
    dynamics = v_mat,
    is_normalized_v = False,
    dt = model.dt,
    neuron_metadata_collection = neuron_metadata_collection,
    fig_axes = [axes[0,0]])

  plot_potentials(
    neuron_names_to_show = neurons_to_observe,
    dynamics = v_normalized_mat,
    is_normalized_v = True,
    dt = model.dt,
    neuron_metadata_collection = neuron_metadata_collection,
    fig_axes = [axes[0,1]])

  # Plot stimulus
  times = np.array(np.arange(0, simul_timesteps * model.dt, model.dt))

  awar_index = neuron_metadata_collection.get_id_from_name("AWAR")
  amps = [time_to_I_ext_fun(time)[awar_index] for time in times]
  ax = axes[1,0]
  ax.plot(times, amps)
  ax.set_title("Stimulus over time for AWAR")
  ax.set_ylabel("Injected current (nA)")
  ax.set_xlabel("Time (s)")

  fig_title = case_name
  fig.suptitle(fig_title, fontsize=16)
  fig.savefig("../local_results/exp8_awa_bruteforce_general/" + fig_title + ".png")
  plt.close(fig)

for case in cases:
  run_one_case(case['case_name'],
    case['init_cond_seed'],
    case['init_cond_amp_nA'],
    case['stim_spec'])

