"""
List of inhibitory neurons.
"""
import numpy as np
# noinspection PyUnresolvedReferences
import project_path  # pylint: disable=unused-import

from model.data_accessor import get_data_file_abs_path
from model.neuron_metadata import NeuronMetadataCollection

def main():
    """
    :return:
    """
    neuron_metadata_collection = NeuronMetadataCollection.load_from_chem_json(\
        get_data_file_abs_path('chem.json'))
    inhibitory_neuron_ids = np.argwhere(\
        np.load(get_data_file_abs_path('emask.npy')).flatten()).flatten()
    inhibitory_neuron_names = []
    for neuron_id in inhibitory_neuron_ids:
        metadata = neuron_metadata_collection.get_metadata(neuron_id)
        inhibitory_neuron_names.append(metadata.name)
    print(sorted(inhibitory_neuron_names))

main()
