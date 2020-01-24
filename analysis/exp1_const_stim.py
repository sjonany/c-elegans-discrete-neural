# Script for experiment 1: Constant stimulus, single parameter at a time.
# This generates a bunch of images at the local_results/exp1/ directory
# This is the automation of the single_param notebook.

# TODO: Migrate notebook code here.

unparsed_cases = [
  "AIY:1",
  "AIY:5",
  "ALM:1",
  "ALM:5",
  "ASE:1",
  "ASE:5",
  "ASH: 1",
  "ASH: 5",
  "AVA:1",
  "AVA:5",
  "AVD:1",
  "AVD:5",
  "AVE:1",
  "AWA:1",
  "AWA:5",
  "AWB:1",
  "AWB:5",
  "AWC:1",
  "AWC:5",
  "AVE:5",
  "IL1:1",
  "IL2:5",
  "IL2:1",
  "IL2:5",
  "PLM:1.4",
  "PLM:5",
  "RIV:1",
  "RIV:5",
  "RMD:1",
  "RMD:5",
  "PLML: 1, PLMR: 5",
  "ASKL: 5, ASKR: 5",
  "ASHR: 1",
  "ASHR: 5",
  "ASHL: 1, ASHR: 5",
  "PLM: 1.4, AVB: 2.3",
  "PLM: 1.4, AVB: 2.3, ALM: 1, AVM: 1",
  "PLM: 1.4, AVB: 2.3, ALM: 5, AVM: 5",
  "PLM: 1.4, AVB: 2.3, RIV: 1",
  "PLM: 1.4, AVB: 2.3, RIV: 5",
  "ASK: 1.4, PLM: 1.4",
  "ASK: 5, PLM: 1.4"
]

cases = []

for unparsed_case in unparsed_cases:
  neuron_to_amp = {}
  neuron_amp_strs = unparsed_case.split(",")
  for neuron_amp_str in neuron_amp_strs:
    neuron_amp = neuron_amp_str.split(":")
    neuron = neuron_amp[0].strip()
    amp = float(neuron_amp[1].strip())
    neuron_to_amp[neuron] = amp
  cases.append(neuron_to_amp)
print(cases)
