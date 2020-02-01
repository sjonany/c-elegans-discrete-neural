"""
Plotting utilities.
"""
import pylab as plt
import numpy as np
from sklearn.decomposition import PCA
from model.neuron_metadata import *
# from .numpy_util import *
from . import numpy_util as np_util

def plot_potentials(neuron_names_to_show, dynamics, is_normalized_v, dt, neuron_metadata_collection):
  """ Plot timeseries charts for the selected neuron names using data from 'dynamics'
  Usage:
    from util.neuron_metadata import *
    neuron_metadata_collection = NeuronMetadataCollection.load_from_chem_json('data/chem.json')
    dynamics = np.load('data/dynamics_fwd_5s.npy')
    plot_saved_dynamics(['PLML', 'PLMR', 'VB01'], dynamics, neuron_metadata_collection)
  Param
    is_normalized_v (bool)
      Set to true if you are plotting the Vth-adjusted traces, instead of the normal V(t).
      This will set change the Y-axis label accordingly.
  """
  dynamics_snapshot_count = dynamics.shape[0]
  num_neurons_to_show = len(neuron_names_to_show)
  fig, axes = plt.subplots(nrows=num_neurons_to_show, ncols=1,
      figsize=(10, 5 * num_neurons_to_show))
  times = np.arange(0, dynamics_snapshot_count * dt, dt)
  for i in range(num_neurons_to_show):
    name = neuron_names_to_show[i]
    id = neuron_metadata_collection.get_id_from_name(name)
    # The neuron ids are already 0-indexed, and is a direct index to dynamics column.
    dynamic = dynamics[:, id]
    
    if num_neurons_to_show == 1:
      ax = axes
    else:
      ax = axes[i]
    ax.plot(times, dynamic)
    ax.set_ylabel(get_v_y_axis_label(is_normalized_v))
    ax.set_xlabel("Time (s)")
    ax.set_title(name)
  return fig

def plot_pcas(dynamics, dt, neuron_metadata_collection):
  """Plot timeseries of PCAs for each of the 3 neuron classes [sensory, motor, interneuron]
  See plot_potentials for usage.
  """
  for neuron_type in [NeuronType.SENSORY, NeuronType.MOTOR, NeuronType.INTERNEURON]:
    relevant_ids = neuron_metadata_collection.get_neuron_ids_by_neuron_type(neuron_type)
    relevant_potentials = dynamics[:, relevant_ids]
    num_pca_comps = 4
    pca = PCA(n_components = num_pca_comps)
    projected_X = pca.fit_transform(relevant_potentials)

    times = np.arange(0, projected_X.shape[0] * dt, dt)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 3))
    for pca_comp in range(num_pca_comps):
      ax.set_title(neuron_type)
      ax.plot(times, projected_X[:,pca_comp], label = "PCA mode {0:d}".format(pca_comp+1))
      ax.legend()

def create_changing_step_bifurcation_plot(
    neurons_to_observe,
    dynamics,
    is_normalized_v,
    step_amps_nA,
    start_amp_nA,
    peak_amp_nA,
    amp_delta_nA, 
    step_duration_timesteps,
    neuron_metadata_collection):
  """
  See changing_step_awa notebook for usage
  """

  # From neuron name to a list of summary statistics of one step of one neuron.
  # Each array element is (prev_I, now_I, minV, maxV, meanV)
  step_results_per_neuron = {}

  for step_amp_i in range(len(step_amps_nA)):
    prev_step_amp_nA = None
    if step_amp_i > 0:
      prev_step_amp_nA = step_amps_nA[step_amp_i-1]
    step_amp_nA = step_amps_nA[step_amp_i]
    timestep_start = int(step_amp_i * step_duration_timesteps)
    timestep_end = step_duration_timesteps + timestep_start - 1
    timestep_stable = int(timestep_end - step_duration_timesteps * 0.2)
    stable_traces_for_step = dynamics[timestep_stable:timestep_end,:]
    for neuron in neurons_to_observe:
      neuron_id = neuron_metadata_collection.get_id_from_name(neuron)
      stable_traces_for_neuron = stable_traces_for_step[:, neuron_id]
      minV = min(stable_traces_for_neuron)
      maxV = max(stable_traces_for_neuron)
      meanV = np.mean(stable_traces_for_neuron)
      
      if neuron not in step_results_per_neuron:
        step_results_per_neuron[neuron] = []
      step_results_per_neuron[neuron].append((prev_step_amp_nA, step_amp_nA, minV, maxV, meanV))

  current_Nas = np.arange(start_amp_nA, peak_amp_nA+amp_delta_nA/2, amp_delta_nA)
  # Key = neuron, value = list of statistics, aligned to current_Nas
  minVs_per_neuron_asc = {}
  maxVs_per_neuron_asc = {}
  meanVs_per_neuron_asc = {}
  minVs_per_neuron_desc = {}
  maxVs_per_neuron_desc = {}
  meanVs_per_neuron_desc = {}

  for neuron in neurons_to_observe:
    for neuron_to_stats in [minVs_per_neuron_asc, maxVs_per_neuron_asc, meanVs_per_neuron_asc, \
                           minVs_per_neuron_desc, maxVs_per_neuron_desc, meanVs_per_neuron_desc]:
      neuron_to_stats[neuron] = [None] * len(current_Nas)


  for neuron in neurons_to_observe:
    # Each array element is (prev_I, now_I, minV, maxV, avgV)
    step_results = step_results_per_neuron[neuron]
    for (prev_I, now_I, minV, maxV, meanV) in step_results:
      min_Vs = minVs_per_neuron_desc[neuron]
      max_Vs = maxVs_per_neuron_desc[neuron]
      mean_Vs = meanVs_per_neuron_desc[neuron]
      if prev_I is None or now_I > prev_I:
        # Ascending case.
        min_Vs = minVs_per_neuron_asc[neuron]
        max_Vs = maxVs_per_neuron_asc[neuron]
        mean_Vs = meanVs_per_neuron_asc[neuron]
      current_Na_aligned_i = np_util.find_nearest_idx(current_Nas, now_I)
      min_Vs[current_Na_aligned_i] = minV
      max_Vs[current_Na_aligned_i] = maxV
      mean_Vs[current_Na_aligned_i] = meanV

  y_label = get_v_y_axis_label(is_normalized_v)

  # Plot the summary statistics per neuron
  for neuron in neurons_to_observe:
    minVs_per_neuron_asc[neuron]
    maxVs_per_neuron_asc[neuron]
    meanVs_per_neuron_asc[neuron]
    
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(current_Nas, minVs_per_neuron_asc[neuron], label="min")
    ax.plot(current_Nas, maxVs_per_neuron_asc[neuron], label="max")
    ax.plot(current_Nas, meanVs_per_neuron_asc[neuron], label="mean")
    ax.set_title("%s ASC" % neuron)
    ax.set_xlim(min(current_Nas), max(current_Nas))
    ax.set_ylabel(y_label)
    ax.set_xlabel("I (nA)")
    ax.legend()
    
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(current_Nas, minVs_per_neuron_desc[neuron], label="min")
    ax.plot(current_Nas, maxVs_per_neuron_desc[neuron], label="max")
    ax.plot(current_Nas, meanVs_per_neuron_desc[neuron], label="mean")
    ax.set_title("%s DESC" % neuron)
    ax.set_xlim(min(current_Nas), max(current_Nas))
    ax.set_ylabel(y_label)
    ax.set_xlabel("I (nA)")
    ax.legend()
    
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(current_Nas, meanVs_per_neuron_asc[neuron], label="asc mean")
    ax.plot(current_Nas, meanVs_per_neuron_desc[neuron], label="desc mean")
    ax.set_title("%s Means of asc and desc" % neuron)
    ax.set_xlim(min(current_Nas), max(current_Nas))
    ax.set_ylabel(y_label)
    ax.set_xlabel("I (nA)")
    ax.legend()
    
    fig, ax = plt.subplots(nrows=1, ncols=1)
    asc_diams = np.array(maxVs_per_neuron_asc[neuron]) - np.array(minVs_per_neuron_asc[neuron])
    # Desc has one entry less than asc, need to remove the last index, which contains None
    desc_diams = np.array(maxVs_per_neuron_desc[neuron][:-1]) - np.array(minVs_per_neuron_desc[neuron][:-1])
    ax.plot(current_Nas, asc_diams, label="asc diam")
    ax.plot(current_Nas[:-1], desc_diams, label="desc diam")
    ax.set_title("%s Diameter of asc and desc" % neuron)
    ax.set_xlim(min(current_Nas), max(current_Nas))
    ax.set_ylabel(y_label)
    ax.set_xlabel("I (nA)")
    ax.legend()

def get_v_y_axis_label(is_normalized_v):
  if is_normalized_v:
    return "f (V-Vth)"
  else:
    return "V (mV)"