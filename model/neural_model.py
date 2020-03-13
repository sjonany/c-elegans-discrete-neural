import numpy as np
from scipy import integrate, linalg, sparse
import pdb
import time
from .neuron_metadata import *
from .data_accessor import *

class NeuralModel:
  """ The C elegans model as described in the paper: "Neural Interactome: Interactive Simulation of a Neuronal System"
  Main reference: https://github.com/shlizee/C-elegans-Neural-Interactome/blob/master/initialize.py
  Usage:
    neuron_metadata_collection = NeuronMetadataCollection.load_from_chem_json(get_data_file_abs_path('chem.json'))
    model = NeuralModel(neuron_metadata_collection)

    # You can tweak parameters before running
    stimulus = {
      "PLML": 1.4,
      "PLMR": 1.4,
    }
    model.set_I_ext_constant_currents(stimulus)

    # This will use your tweaked parameters to read files and precompute some values.
    model.init()
    (v_mat, s_mat, v_normalized_mat) = model.run(100)

    For an end-to-end example, please see milestone_responsive.ipynb
  """
  def __init__(self, neuron_metadata_collection, C=0.015, Gc=0.1, ggap=1.0, gsyn=1.0):
    self.neuron_metadata_collection = neuron_metadata_collection
    # Number of neurons
    self.N = neuron_metadata_collection.get_size()

    self.init_conds = 10**(-4)*np.random.normal(0, 0.94, 2*self.N)

    self.dt = 0.01

    # I_i(t), where i is a neuron index
    self.I_ext_t = lambda t: np.zeros(self.N)

    # An array of time values where I_ext_t changes, and so need to be re-invoked
    # These are actual t values in the dynamical system, and so can be non-integers
    # Performance optimization. 15% speedup. See
    # https://github.com/sjonany/c-elegans-discrete-neural/commit/cfdfbab3464946872b5fc6e17e04a4885cca7550
    self.t_changes_I_ext = [0]

    # The current I_i vector at a specific time
    self.cur_I_ext = None

    # Cell membrane capacitance. Default: 1.5 pF / 100 = 0.015 arb (arbitrary unit)
    self.C = C

    # Cell membrane conductance, for calculating I_leak. Default: 10pS / 100 = 0.1 arb
    self.Gc = Gc

    # Leakage potential (mV)
    self.Ec = -35.0

    # ggap Default: 100pS / 100 = 1 arb
    self.ggap = ggap

    # gsyn Default: 100pS / 100 = 1 arb
    self.gsyn = gsyn

    # Synaptic activity
    # Synaptic activity's rise time
    self.ar = 1.0/1.5
    # Synaptic activity's decay time 
    self.ad = 5.0/1.5
    # Width of the sigmoid (mv^-1)
    self.B = 0.125

  def set_I_ext(self, time_to_I_ext_fun, t_changes_I_ext):
    """
    Set the injected currents, which can vary over time.
    time_to_I_ext_fun specs:
        Input: t
        Output: A vector of scalars I_ext_i, one for each neuron.
            The scalars should be in nA.
            The neuron ids are based off neuron_metadata_collection.
    t_changes_I_ext:
        An array of time values where I_ext(t) changes, and needs to be re-invoked.
    """
    # Need to rescale from nA to arbs
    self.I_ext_t = lambda t: NeuralModel.nA_to_arbs(np.array(time_to_I_ext_fun(t)))
    self.t_changes_I_ext = t_changes_I_ext

  def set_I_ext_constant_currents(self, neuron_name_to_current_nA):
    """
    Helper function on top of set_I_ext if you just want constant currents.
    Unlike set_I_ext_arbs, the current units are in nA.
    neuron_name_to_current_nA: A dictionary
        key: string like "AWAL"
        value: constant injected current like 2.3 nA
    """
    neuron_id_to_current_nA = [0.0] * self.N
    for neuron, current_nA in neuron_name_to_current_nA.items():
        neuron_id = self.neuron_metadata_collection.get_id_from_name(neuron)
        neuron_id_to_current_nA[neuron_id] = current_nA
    self.set_I_ext(lambda t: neuron_id_to_current_nA, [0])

  @staticmethod
  def nA_to_arbs(nA):
    return nA * 10000

  def init(self):
    """
    Initialize parameters according to Kim et al., 2019.
    """
    # Gap junctions. Total conductivity of gap junctions, where total conductivity = #junctions * ggap.
    # An N x N matrix were Gg[i][j] = from neuron j to i.
    self.Gg = np.load(get_data_file_abs_path('Gg.npy')) * self.ggap

    # Synaptic junctions. Total conductivity of synapses, where total conductivity = #junctions * gsyn.
    # An N x N matrix were Gs[i][j] = from neuron j to i.
    self.Gs = np.load(get_data_file_abs_path('Gs.npy')) * self.gsyn

    # E. Reversal potential for each neuron, for calculating I_syn
    # 0 mV for synapses going from an excitatory neuron, -48 mV for inhibitory.
    # N-element array that is 1.0 if inhibitory.
    is_inhibitory = np.load(get_data_file_abs_path('emask.npy'))
    self.E = np.reshape(-48.0 * is_inhibitory, self.N)

  def init_kunert_2017(self):
    """
    Change parameters to Kunert et al., 2017: "Multistability ..."
    See section 2.2 model parameters.
    The parameters are slightly different from Kimin et al., 2019
    See init() for more documentation.
    Remember that pS to arb (unit used by the model), you divide by 100.
    """
    self.ggap = 1.0
    self.gsyn = 1.0
    self.Gc = 0.1
    self.Ec = -35.0
    self.C = 0.01
    self.Gg = np.load(get_data_file_abs_path('Gg.npy')) * self.ggap
    self.Gs = np.load(get_data_file_abs_path('Gs.npy')) * self.gsyn

    # ar and ad are given in Kunert et al., 2014, which strangely enough has different
    # values for the previous params, and more in line with Kim et al., 2019's.
    self.ar = 1.0 
    self.ad = 5.0
    is_inhibitory = np.load(get_data_file_abs_path('emask.npy'))
    self.E = np.reshape(-45.0 * is_inhibitory, self.N)


  def compute_Vth(self):
    """
    Vth computation that I wrote from scratch, and that matches my math derivations more.
    Validations:
    - Ran interactome code and printed out their Vth.
      Vth 1 30 100, sum = -18.3342063908 -6.2498993778 -3.61729185518 -1194.25458719
    - compute_Vth(), this method, produces very similar results as well.
      Vth 1 30 100, sum = -18.334206390780512 -6.249899377800608 -3.6172918551786215 -1194.2545871854845
    - L2 norm of compute_Vth() and interactome's:  3.907985046680551e-14
    """
    b1 = -np.tile(self.Gc * self.Ec, self.N)
    s_eq = self.compute_s_eq()
    b3 = -s_eq * (self.Gs @ self.E)

    m1 = -self.Gc * np.identity(self.N)
    # N x 1, where each item is a row sum
    Gg_row_sums = self.Gg.sum(axis = 1)
    # m2 is a diagonal matrix with the negative row sums as the values
    m2 = - np.diag(Gg_row_sums)
    Gs_row_sums =  self.Gs.sum(axis = 1)
    m3 = - s_eq * np.diag(Gs_row_sums)
    # I think paper is missing m4. It shouldn't be the case that A is a completely diagonal matrix.
    # However, interactome github code seems to have done this correctly.
    # Our implementation is mathematically equivalent to the github code.
    m4 = self.Gg

    A = m1 + m2 + m3 + m4
    # TODO: Create a mode where you don't use moving Vth.
    # b = b1 + b3
    b = b1 + b3 - self.cur_I_ext
    self.A = A
    self.Vth = np.reshape(linalg.solve(A, b), self.N)
  
  def compute_s_eq(self):
    """
    The equilibrium values for all si's are the same, no matter I_ext. Just set phi = 0.5.
    """
    return self.ar / (self.ar + 2 * self.ad)

  def compute_standard_equilibrium(self):
    """
    Compute the standard equilibrium, where for a given I_ext, if you start close to here,
    you will stay here forever (assuming this is a stable fixed point -- but sometimes it's not)
    Useful for analysis, but not for model running.
    """
    self.compute_Vth()
    s_eq = self.compute_s_eq()
    return np.append(self.Vth, np.array([s_eq] * self.N))

  def update_cur_I_ext_and_Vth(self, t):
    if len(self.t_changes_I_ext) > 0 and t >= self.t_changes_I_ext[0]:
        self.cur_I_ext = self.I_ext_t(t)
        # Get the next change time
        self.t_changes_I_ext.pop(0)
        # Vth depends on I_ext.
        # If have time, make this computation more efficient like Kim et al., 2019's code.
        self.compute_Vth()


  def dynamic(self, t, state_vars):
    """Dictates the dynamics of the system.
    """
    v_arr, s_arr = np.split(state_vars, 2)
    self.update_cur_I_ext_and_Vth(t)

    # I_leak
    I_leak = self.Gc * (v_arr - self.Ec)

    # I_gap = sum_j G_ij (V_i - V_j) = V_i sum_j G_ij - sum_j G_ij V_j
    # The first term is a point-wise multiplication of V and G's squashed column.
    # The second term is matrix multiplication of G and V
    I_gap = self.Gg.sum(axis = 1) * v_arr - self.Gg @ v_arr
    
    # I_syn = sum_j G_ij s_j (V_i - E_j) = V_i sum_j G_ij s_j - sum_j G_j s_j E_j
    # First term is a point-wise multiplication of V and (Matrix mult of G and s)
    # Second term is matrix mult of G and (point mult of s_j and E_j)
    I_syn = v_arr * (self.Gs @ s_arr) - self.Gs @ (s_arr * self.E)

    dV = (-I_leak - I_gap - I_syn + self.cur_I_ext) / self.C
    phi = 1.0/(1.0 + np.exp(-self.B*(v_arr - self.Vth)))
    syn_rise = self.ar * phi * (1 - s_arr)
                          
    syn_drop = self.ad * s_arr
    dS = syn_rise - syn_drop
    return np.concatenate((dV, dS))

  def get_normalized_v_arr(self, v_arr):
    """The paper performs analysis on this normalized v_arr.
    """
    vth_adjusted = v_arr - self.Vth
    vmax = 500

    # tanh: Similar to sigmoid, but squashes to between -1 and 1.
    # So, below readjusts value to range from -500 to 500.
    return vmax * np.tanh(np.divide(vth_adjusted, vmax)) 

  def run(self, num_timesteps):
    """Create initial conditions, then simulate dynamics num_timesteps times.
    Args:
      num_timesteps (int): The number of simulation timesteps to run for. Each timestep is dt = 0.01 second long.
    Returns:
      v_mat (num_timesteps x N): Each column is a voltage timeseries of a neuron. 
      s_mat (num_timesteps x N): Each column is an activation timeseries of a neuron's synaptic current.
      v_normalized_mat (num_timesteps x N): v_mat, but normalized just like the exported dynamics file from Interactome.
        Note that interactome's exported data starts from timestep 50 onwards, so make sure to truncate first 50 if you
        want to compare. See milestone_responsive.ipynb for how to do this.
    """

    N = self.N
    dt = self.dt

    # The variables to store our complete timeseries data.
    v_mat = []
    s_mat = []
    v_normalized_mat = []

    dyn = integrate.ode(self.dynamic).set_integrator('vode', atol = 1e-3,
        min_step = dt*1e-6, method = 'bdf', with_jacobian = True,
        max_step = dt)
    dyn.set_initial_value(self.init_conds , 0)

    start_time = time.time()
    for t in range(num_timesteps):
      if t % 100 == 0:
        print("Timestep %d out of %d" % (t, num_timesteps))
      dyn.integrate(dyn.t + dt)
      v_arr = dyn.y[:N]
      s_arr = dyn.y[N:]
      v_normalized_arr = self.get_normalized_v_arr(v_arr)
      v_mat.append(v_arr)
      s_mat.append(s_arr)
      v_normalized_mat.append(v_normalized_arr)
    print("Total runtime = %.2fs" % (time.time() - start_time))
    return np.array(v_mat), np.array(s_mat), np.array(v_normalized_mat)