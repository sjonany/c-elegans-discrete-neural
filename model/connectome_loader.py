"""
Utility methods for loading various connectomes.
"""
import csv
import numpy as np

from model.data_accessor import get_data_file_abs_path
from model.neuron_metadata import NeuronMetadataCollection


def load_gap_connectome_varshney():
    """
    Get gap junction and chemical synapse connectivity matrix from Varshney et al., 2011.
    The returned value is a tuple of (gap junction matrix, chem matrix)
    """
    return np.load(get_data_file_abs_path('Gg.npy')), np.load(get_data_file_abs_path('Gs.npy'))


def load_connectome_cook():
    """
    Get gap junction and chemical synapse connectivity matrix from Cook et al., 2019.
    The returned value is a tuple of (gap junction matrix, chem matrix)
    """
    # Cook has extra neurons. It's on purpose that we use Varshney's list of neurons.
    neuron_metadata_collection = \
        NeuronMetadataCollection.load_from_chem_json(get_data_file_abs_path('chem.json'))
    conn_spec_to_weight_gap, conn_spec_to_weight_chem = load_connectome_dict_cook()
    return (build_connectome_matrix_from_dict(conn_spec_to_weight_gap, neuron_metadata_collection),
            build_connectome_matrix_from_dict(conn_spec_to_weight_chem, neuron_metadata_collection))


def build_connectome_matrix_from_dict(conn_spec_to_weight, neuron_metadata_collection):
    """
    Helper method to convert a connectome in dictionary form to matrix form.
    """
    N = neuron_metadata_collection.get_size()
    mat = np.zeros((N, N))

    for conn_spec, weight in conn_spec_to_weight.items():
        source, target = conn_spec
        source_id = neuron_metadata_collection.get_id_from_name(source)
        target_id = neuron_metadata_collection.get_id_from_name(target)
        if source_id < 0 or target_id < 0:
            # Skip. Cook has extra pharyngeal neurons.
            # See https://docs.google.com/document/d/14KvRBBwdQCg6zsKXNWArELXbpAEcRHKj2LAZfwJN_ns/edit#heading=h.7sdtrhgj2ujx
            continue
        # The existing Gg[i][j] means from neuron j to i.
        mat[target_id, source_id] = weight
    return mat


def load_connectome_dict_cook():
    """
    Get gap junction and chemical synapse connectivity matrix from Cook et al., 2019.
    The returned value is a tuple of (conn_spec_to_weight_gap, conn_spec_to_weight_chem)
    Each conn_spec_to_weight is a dictionary with key of conn_spec to weight
    conn_spec is a tuple (source neuron, target neuron)

    Usage:
      conn_spec_to_weight_gap, conn_spec_to_weight_chem = load_connectome_dict_cook()
      # This gives gap junction weight from ASHL to ASHR
      conn_spec_to_weight_gap[('ASHL', 'ASHR')]
    """
    connectome_file = 'herm_full_edgelist.csv'

    # key = conn_spec = (from, to)
    # value = total weight
    conn_spec_to_weight_chem = {}
    conn_spec_to_weight_gap = {}
    with open(get_data_file_abs_path(connectome_file), newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            source = row['Source'].upper().strip()
            target = row['Target'].upper().strip()
            weight = row['Weight'].upper().strip()
            conn_type = row['Type'].upper().strip()
            conn_spec = (source, target)

            conn_spec_to_weight = None
            if conn_type == "CHEMICAL":
                conn_spec_to_weight = conn_spec_to_weight_chem
            elif conn_type == "ELECTRICAL":
                conn_spec_to_weight = conn_spec_to_weight_gap
            else:
                raise Exception("Invalid connection type: " + conn_type)

            if conn_spec in conn_spec_to_weight:
                raise Exception(
                    "Duplicate entry exists for %s. Previous value is %d, new value is %d" % \
                    (conn_type, conn_spec_to_weight[conn_spec], weight))
            conn_spec_to_weight[conn_spec] = weight
    return (conn_spec_to_weight_gap, conn_spec_to_weight_chem)
