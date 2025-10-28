# Encoding Dictionaries: Visualize and Generate Designs

Encoding dictionaries let you specify concrete signal encodings and build a design directly from them.  
They’re useful for sharing designs between **GENIAL** and **Flowy**, or for reproducing specific encodings end-to-end.

---

## What You Can Do

- **Visualize** an encoding (input or output) from a file.  
- **Generate** a new design whose LUT is defined by your encoding dictionary.

---

## Accepted Input Formats

You can pass any of these file types as `--enc_dict_path` or `-f`:

### JSON (recommended)

**Minimal format** (input required, output optional):

```json
{
  "input":  { "-2": "10", "-1": "11", "0": "00", "1": "01" },
  "output": { "-2": "1110", "-1": "1111", "0": "0000", "1": "0001", "2": "0010", "4": "0100" }
}
```

**Flowy-style format** is also recognized:

```json
{
  "encoding": {
    "data": {
      "input":  { "data": { "-2": "10", "-1": "11", "0": "00", "1": "01" } },
      "output": { "data": { "-2": "1110", "-1": "1111", "0": "0000", "1": "0001", "2": "0010", "4": "0100" } }
    }
  }
}
```

### Verilog (`.v`) or compressed Verilog (`.v.bz2`) with inline comments

```verilog
// input encoding
// -2 -> 10
// -1 -> 11
//  0 -> 00
//  1 -> 01
// output encoding
// -2 -> 1110
// -1 -> 1111
//  0 -> 0000
//  1 -> 0001
//  2 -> 0010
//  4 -> 0100
```

**Notes**
- Input encoding (`input`) is required to build a design. Output encoding (`output`) is optional; if omitted, GENIAL derives it from the experiment configuration (encoding type + bitwidth).  
- Keys are integer values; values are bit-strings. Make sure bit-string widths match the configured bitwidths.

---

## Visualize an Encoding

Use the `visualize_encoding` CLI to produce a heatmap image saved next to your file as `encoding_representation.png`.

**Command:**
```bash
visualize_encoding -f <path>/encoding_dict.json -b <bitwidth> -t <input|output>
```

**Example (8-bit input encoding visualization):**
```bash
visualize_encoding -f sme_encoding.json -b 8 -t input
```

The tool also works with `.v` and `.v.bz2` files containing inline encoding comments.

---

## Generate a Design From an Encoding Dict

The `generate_design` CLI builds a new design using your encoding dict and writes it to the experiment’s output directory.  
It automatically assigns the next available design number.

### Prerequisites

- Install and activate the environment (see README “Quick Start”).  
- Ensure `.env` is configured and sourced so that `SRC_DIR` and `WORK_DIR` are set.

### Command

```bash
generate_design   --experiment_name <template_configuration_to_use>   --output_dir_name <anything>   --enc_dict_path <path>/encoding_dict.json
```

**Python module equivalent:**

```bash
python -m genial.tools.generate_design_from_encoding_dict --enc_dict_path <path>/encoding_dict.json
```

### Where the Results Go

```
$WORK_DIR/output/<experiment_name>/<output_dir_name>/generation_out/res_<design_number>/hdl/mydesign_comb.v
```

### Tips

- Only the `input` encoding is strictly required.  
  When `output` is absent, GENIAL derives it based on the experiment settings (`output_encoding_type`, `output_bitwidth`).  
- For adders, outputs outside the input range are saturated to min/max representable values.  
  For multipliers, all product values must be present in `out_enc_dict` when provided.

---

## Minimal Working Example (2-bit Multiplier)

Use the built-in experiment `multiplier_2bi_4bo_permuti_allcells_notech_fullsweep_only` (2-bit inputs, 4-bit outputs, two’s complement) and a tiny input encoding dict:

1. Create a file `examples/mult2_tc_input.json`:

```json
{
  "input": {
    "-2": "10",
    "-1": "11",
    "0":  "00",
    "1":  "01"
  }
}
```

2. Visualize the input encoding (optional):

```bash
visualize_encoding -f examples/mult2_tc_input.json -b 2 -t input
```

3. Generate the design:

```bash
generate_design   --experiment_name multiplier_2bi_4bo_permuti_allcells_notech_fullsweep_only   --output_dir_name demo_from_dict   --enc_dict_path examples/mult2_tc_input.json
```

The output netlist will be located at:

```
.../generation_out/res_<N>/hdl/mydesign_comb.v
```

under the experiment output directory.

---

## Troubleshooting

- **“Input encoding missing”** → ensure your file contains an `input` mapping (or valid inline Verilog comments).  
- **“Bitwidth mismatch in visualization”** → use the correct `-b/--bitwidth` matching the bit-strings.  
- **“Where is the output saved?”** → see *Where the Results Go*; the CLI also logs the generated path.

---

## Related Tools / Sources

- **CLI wrappers (installed with the package):**
  - `visualize_encoding` → `genial.tools.plot_encoding_from_file:main_cli`
  - `generate_design`   → `genial.tools.generate_design_from_encoding_dict:main_cli`

- **Internals used:**
  - Parser → `src/genial/experiment/file_parsers.py:get_encoding_dict_from_file`
  - Generator → `src/genial/experiment/task_generator.py:DesignGenerator`
