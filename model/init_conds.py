"""
Utility methods for setting the neural model at interesting initial conditions
"""
import numpy as np

def generate_init_cond_with_seed(model, neuron, current_nA, seed):
  """
  Helper method for generating initial conditions for the model that would
  lead to interesting long term behaviors.
  We add gaussian noise to the standard equilibrium (a.k.a Vth and seq) based on seed.
  For interesting parameter values, exp8a_AWA_SPIKE_BRUTEFORCE_FIXED_POINTS for usage.
  --- Code
  	model.init_conds = init_conds.generate_init_cond_with_seed(
  	model, key_neuron, current_nA, init_cond_seed)
  ---
  """
  model.set_I_ext_constant_currents({neuron: current_nA})
  model.init_kunert_2017()
  model.update_cur_I_ext_and_Vth(0)
  std_equi_V = model.Vth
  s_eq = round(model.ar / (model.ar + 2 * model.ad), 4)
  std_equi = np.append(std_equi_V, [s_eq] * model.N)

  np.random.seed(seed)
  init_conds = std_equi + 10**(-4)*np.random.normal(0, 0.94, 2*model.N)
  return init_conds