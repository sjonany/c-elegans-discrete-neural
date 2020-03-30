# Resources, and their descriptions.
## Model files
### chem.json
* **Usage**. We only use the neuron name and type (Sensory, motor, interneuron).
  * Sensory - blue - group 1 - 84 - E.g. PLML 
  * Motor - red - group 2 - 109 - E.g. AS01, VD01
  * Interneuron - green - group 3 - 86 - E.g. AVAL, AVAR
* **Source**. From https://github.com/shlizee/C-elegans-Neural-Interactome/blob/master/chem.json
### Gg.npy, Gs.npy
* **Usage**. Number of synaptic and gap junctions between pairs of neurons. N x N matrix of integers.
* **Source**. From https://github.com/shlizee/C-elegans-Neural-Interactome/blob/master/Gg.npy 

### emask.npy
* **Usage**. Whether or not neuron in excitatory or inhibitory. An N-element array of 0-1's. 1 if inhibitory.
* **Source**. From https://github.com/shlizee/C-elegans-Neural-Interactome/blob/master/emask.npy 

### herm_full_edgelist.csv
* **Usage**. Cook et al., 2019's connectome, meant to be an update over Varshney et al., 2011.
* **Source**. https://wormwiring.org/series/ > Combined Anterior + Posterior (N2U + JSE) > Edge list.