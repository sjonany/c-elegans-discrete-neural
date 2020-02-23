import numpy as np
import pylab as plt
import project_path
from model.data_accessor import get_data_file_abs_path
from model.neuron_metadata import *
from model.neural_model import NeuralModel
import model.init_conds as init_conds
from util.plot_util import *

"""
Parameters
"""
# How many seconds to run each step for. This should be long enough for stability to be observed.
step_duration_s = 4.5
num_step_increase = 3
key_neuron = "AWAR"
neurons_to_stimulate = [key_neuron]
neurons_to_observe = [key_neuron]
# init_cond_seed, start_amp_nA, amp_delta_nA

#amp_delta_nAs = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]
amp_delta_nAs = list(0.002 * np.array(list(range(1,10))))
print(amp_delta_nAs)
cases = [
  #*[(0, 3, amp_delta_nA) for amp_delta_nA in amp_delta_nAs],
  #*[(3, 3, amp_delta_nA) for amp_delta_nA in amp_delta_nAs],
  #*[(5, 3, amp_delta_nA) for amp_delta_nA in amp_delta_nAs],
  #*[(0, 6, amp_delta_nA) for amp_delta_nA in amp_delta_nAs],
  #*[(1, 6, amp_delta_nA) for amp_delta_nA in amp_delta_nAs],
  #*[(3, 6, amp_delta_nA) for amp_delta_nA in amp_delta_nAs],
  #*[(0, 9, amp_delta_nA) for amp_delta_nA in amp_delta_nAs],
  #*[(1, 9, amp_delta_nA) for amp_delta_nA in amp_delta_nAs],
  #*[(2, 9, amp_delta_nA) for amp_delta_nA in amp_delta_nAs]
  *[(11, 6.7, amp_delta_nA) for amp_delta_nA in amp_delta_nAs]
]


neuron_metadata_collection = NeuronMetadataCollection.load_from_chem_json(get_data_file_abs_path('chem.json'))
N = neuron_metadata_collection.get_size()
model = NeuralModel(neuron_metadata_collection)
neurons_to_stimulate = [key_neuron]

def gen_plot_for_one_init_cond(init_cond_seed, start_amp_nA, amp_delta_nA):
  model = NeuralModel(neuron_metadata_collection)
  peak_amp_nA = start_amp_nA + amp_delta_nA * num_step_increase
  step_amplitudes_nA = np.arange(start_amp_nA, peak_amp_nA-amp_delta_nA/2, amp_delta_nA)
  initial_conds = init_conds.generate_init_cond_with_seed(
      model, key_neuron, start_amp_nA, init_cond_seed)
  step_duration_timesteps = int(step_duration_s / model.dt)
  simul_timesteps = step_duration_timesteps * len(step_amplitudes_nA)

  model.init_conds = initial_conds
  # These are timesteps when I_ext changes
  num_step_values = len(step_amplitudes_nA)
  t_changes_I_ext = np.array(range(num_step_values)) * step_duration_s
  t_changes_I_ext = t_changes_I_ext.tolist()

  def time_to_I_ext_fun(t):
    amp = step_amplitudes_nA[int(t / step_duration_s)]
    cur_I_ext = np.zeros(N)
    for neuron in neurons_to_stimulate:
      neuron_id = neuron_metadata_collection.get_id_from_name(neuron)
      cur_I_ext[neuron_id] = amp
    return cur_I_ext
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

  create_changing_step_bifurcation_plot(
      neurons_to_observe = neurons_to_observe,
      dynamics = v_mat,
      is_normalized_v = False,
      step_amps_nA = step_amplitudes_nA,
      start_amp_nA = start_amp_nA,
      peak_amp_nA = peak_amp_nA,
      amp_delta_nA = amp_delta_nA, 
      step_duration_timesteps = step_duration_timesteps,
      neuron_metadata_collection = neuron_metadata_collection,
      fig_axes = {key_neuron: [axes[1,0], None, None, None]})

  create_changing_step_bifurcation_plot(
      neurons_to_observe = neurons_to_observe,
      dynamics = v_normalized_mat,
      is_normalized_v = True,
      step_amps_nA = step_amplitudes_nA,
      start_amp_nA = start_amp_nA,
      peak_amp_nA = peak_amp_nA,
      amp_delta_nA = amp_delta_nA, 
      step_duration_timesteps = step_duration_timesteps,
      neuron_metadata_collection = neuron_metadata_collection,
      fig_axes = {key_neuron: [axes[1,1], None, None, None]})
  fig_title = "start_nA=%.1f,seed=%d,step_delta_nA=%.3f" % (start_amp_nA, init_cond_seed, amp_delta_nA)
  fig.suptitle(fig_title, fontsize=16)
  fig.savefig("../local_results/exp8_awa_bruteforce/" + fig_title + ".png")
  plt.close(fig)

for (init_cond_seed, start_amp_nA, amp_delta_nA) in cases:
  gen_plot_for_one_init_cond(init_cond_seed, start_amp_nA, amp_delta_nA)

