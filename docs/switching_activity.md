# Switching Activity and Capacitance Model

The analyzer estimates dynamic cost by counting bit flips on each net and weighting
those transitions by an estimate of the load driven by the net.

## Switching vectors

For each trace the analyzer builds a *switching vector* where each entry is `1`
whenever the value of the corresponding wire differs from the previous sample.
The first sample is ignored so the initial state does not count as a toggle.

```python
from genial.experiment.task_analyzer import Analyzer
switch_vectors, totals, wires = Analyzer.get_switching_vectors(db)
```

## Capacitance weights

Transitions are optionally scaled by the fanout capacitance of the net.  The
`get_fanout_wires_n_depth` helper parses the synthesized Verilog netlist and
sums the input pin capacitance of all fanout cells.  A mapping from cell name to
its total input capacitance can be generated from a Liberty file using the
stand‑alone script:

```bash
scripts/liberty_to_capacitance.py <library.lib[.gz]> cell_caps.json
```

Pass the resulting JSON as the `cell_cost_model` argument when calling
`get_fanout_wires_n_depth`.

Applying the weights to the switching vectors yields a simple dynamic power
proxy:

```python
sv_weighted, missing = Analyzer.apply_weights_to_switch_vectors(switch_vectors, fanout_caps)
proxy = sv_weighted.sum().sum()
```

## Selecting the cost model

`task_analyzer` accepts `--cell_cost_model` to choose how fanout weights are
built.  Options are:

* `transistor` – use gate transistor counts (default)
* `capacitance` – sum of input pin capacitances from the technology Liberty
* `capacitance_calibrated` – capacitances after calibration
* `none` – do not weight toggles

The `--technology` argument selects which library directory under
`resources/libraries/<tech>` is searched for the corresponding JSON files
(`transistor_count.json`, `capacitance.json`, `capacitance_calibrated.json`).
If a requested model is missing for the chosen technology, the analyzer aborts
with a clear error message.

## Calibration

The `scripts/swact_calibration.py` script fits a linear regression between the
weighted switching activity and measured power.  Provide a CSV file with columns
`swact` and `power` and it prints the fitted coefficients `a` and `b` for
`power ≈ a * swact + b`.

```bash
scripts/swact_calibration.py data.csv
```
