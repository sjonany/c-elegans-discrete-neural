"""
Plotting utilities.
"""
import pylab as plt
import numpy as np
from sklearn.decomposition import PCA
from model.neuron_metadata import *

def plot_potentials(neuron_names_to_show, dynamics, dt, neuron_metadata_collection):
  """ Plot timeseries charts for the selected neuron names using data from 'dynamics'
  Usage:
    from util.neuron_metadata import *
    neuron_metadata_collection = NeuronMetadataCollection.load_from_chem_json('data/chem.json')
    dynamics = np.load('data/dynamics_fwd_5s.npy')
    plot_saved_dynamics(['PLML', 'PLMR', 'VB01'], dynamics, neuron_metadata_collection)
  """
  dynamics_snapshot_count = dynamics.shape[0]
  num_neurons_to_show = len(neuron_names_to_show)
  fig, axes = plt.subplots(nrows=num_neurons_to_show, ncols=1,
      figsize=(10, 4 * num_neurons_to_show))
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
