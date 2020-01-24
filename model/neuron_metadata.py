from enum import Enum
from itertools import chain
import json

class NeuronType(Enum):
  SENSORY = 1
  MOTOR = 2
  INTERNEURON = 3 

class NeuronMetadataCollection:
  """A collection of all the neuron names and types.
  Usage:
    collection = NeuronMetadataCollection.load_from_chem_json('data/chem.json')
    print(collection.get_metadata(222))
    This will get you metadata for neuron id 222.
  """

  def __init__(self, id_to_metadata):
    """
    Paramaters
    ----------
    id_to_metadata : dictionary
      dictionary from id (int) to NeuronMetadata. E.g. 0 -> <"PLML", NeuronType.SENSORY>
    """
    self.id_to_metadata = id_to_metadata
    self.name_to_id = {}
    for (id, metadata) in id_to_metadata.items():
      self.name_to_id[metadata.name] = id

  def get_metadata(self, id):
    return self.id_to_metadata[id]

  def get_id_from_name(self, name):
    return self.name_to_id[name]

  def get_size(self):
    return len(self.id_to_metadata)

  def get_neuron_ids_by_neuron_type(self, neuron_type):
    ids = []
    for (id, metadata) in self.id_to_metadata.items():
      if metadata.neuron_type == neuron_type:
        ids.append(id)
    return ids

  """
  Usage:
    create_lr_names_from_base(["PLM", "ALM"])
  Output:
    ["PLML", "PLMR", "ALML", "ALMR"]
  """
  @staticmethod
  def create_lr_names_from_base(base_names):
    return list(chain.from_iterable(
      (base_name + "L", base_name + "R")
      for base_name in base_names))

  @staticmethod
  def json_group_to_neuron_type(group):
    if group == 1:
        return NeuronType.SENSORY
    elif group == 2:
        return NeuronType.MOTOR
    elif group == 3:
        return NeuronType.INTERNEURON
    else:
        raise Exception('Invalid neuron group {}'.format(group))

  @staticmethod
  def neuron_type_to_str(neuron_type):
    if neuron_type == NeuronType.SENSORY:
      return "SENSORY"
    elif neuron_type == NeuronType.MOTOR:
      return "MOTOR"
    elif neuron_type == NeuronType.INTERNEURON:
      return "INTERNEURON"
    else:
        raise Exception('Invalid neuron type {}'.format(neuron_type))


  @staticmethod
  def load_from_chem_json(path_to_chem_json):
    with open(path_to_chem_json) as f:
      chem_json = json.loads(f.read())
    id_to_metadata = {}
    for node_info in chem_json['nodes']:
      id = node_info['index']
      name = node_info['name']
      neuron_type = NeuronMetadataCollection.json_group_to_neuron_type(node_info['group'])
      id_to_metadata[id] = NeuronMetadata(id, name, neuron_type)
    return NeuronMetadataCollection(id_to_metadata)


class NeuronMetadata:
  """A single neuron's metadata."""
  
  def __init__(self, id, name, neuron_type):
    """
    Parameters
    ----------
    id : int
    name : str
      E.g. PLML
    neuron_type: NeuronType
      E.g. NeuronType.SENSORY
    """
    self.id = id
    self.name = name
    self.neuron_type = neuron_type

  def __str__(self):
    return "{} {} {}".format(self.id, self.name, self.neuron_type)