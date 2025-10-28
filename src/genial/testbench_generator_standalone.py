from pathlib import Path
from string import Template
import re
import argparse

from genial.utils.utils import extract_int_string_from_string


def _clean_output_pin_names(text):
    """Remove the unsafe patterns from the output pin names"""
    return text.replace("\\", "").replace(" ", "").replace(".", "_")


def get_genered_design_dir_path(generation_out_dir, design_number):
    """Returns the path to the generated design associated with the design_number"""
    return generation_out_dir / f"res_{design_number}/hdl"


def get_genered_design_file_path(generation_out_dir, design_number):
    """Returns the path to the generated design associated with the design_number"""
    return get_genered_design_dir_path(generation_out_dir, design_number) / "mydesign_comb.v"


def get_wire_list(synthed_design_path, remove_dangling_wires, clean_output_pin_names=True, tool_name="design_compiler"):
    """Parses the synthesized design file `synthed_design_path` and returns the list of internal wires.
    If `remove_dangling_wires` is True, it excludes the wires that do not ave fanout cells from the returned list."""
    # Read design
    with open(synthed_design_path) as f:
        design = f.readlines()

    if tool_name == "yosys":
        # Extract all wires
        wire_list = []
        wire_pattern = re.compile(r"wire _[0-9]*_ ?;")
        re.compile(wire_pattern)
        is_reg = False
        for line in design:
            # Get all internal wires
            if line.strip().startswith("wire"):
                match = wire_pattern.search(line)
                if match:
                    wire_name = line.split("_")[1]
                    wire_list.append("_" + wire_name + "_")

            # Deal with registers
            if is_reg:
                if ".D(" in line:
                    wire_name = line.split("(")[1].split(")")[0]
                    if clean_output_pin_names:
                        wire_name = _clean_output_pin_names(wire_name)
                    # For some designs (namely encoders), it is possible that an input wire is directly connected to output register
                    # In this case, we don't add it anyway because the switching activity measured in input is already counted
                    # Add unconventional named wires to wire list
                    if wire_name not in wire_list:
                        wire_list.append(wire_name)

                if remove_dangling_wires:
                    if ".Q_N(" in line:
                        wire_name = line.split("(")[1].split(")")[0]
                        # Remove dangling wires
                        wire_index = wire_list.index(wire_name)
                        del wire_list[wire_index]

            # Filter for register instances
            if "__reg" in line:
                is_reg = True
            if is_reg and ");" in line:
                is_reg = False

    elif tool_name == "design_compiler":
        for line in design:
            if "wire" in line and ";" in line:
                wires = line.replace("wire", "").replace(";", "").replace(",", "")
                wire_list = wires.split(" ")
                print(f"Extracted list of wires is:")
                print(wire_list)
                break

    return wire_list


def extract_encodings(verilog_module_filepath, reverted=False):
    """File function reads the encoding found int the verilog file."""
    # Encoding pattern is: `// value -> representation`
    encoding_pattern = re.compile(r"// -?\d+ -> [0-1]+")
    encodings = dict()
    enc_type = None
    vals_list = []
    repr_list = []
    with open(verilog_module_filepath, "r") as f:
        for line in f:
            if line.startswith("//") and "encoding" in line.lower():
                if enc_type is not None:
                    # Store already read encodings
                    if reverted:
                        encodings[enc_type] = {enc: int(val) for val, enc in zip(vals_list, repr_list)}
                    else:
                        encodings[enc_type] = {int(val): enc for val, enc in zip(vals_list, repr_list)}

                # Encoding lists initialization
                vals_list = []
                repr_list = []
                if "input" in line.lower() or "in_enc_dict" in line.lower():
                    enc_type = "input"
                elif "output" in line.lower() or "out_enc_dict" in line.lower():
                    enc_type = "output"
                else:
                    enc_type = "both"

            match = encoding_pattern.search(line)
            if match:
                value, repre = match.group(0).split("->")
                value = value.strip("/").strip()
                repre = extract_int_string_from_string(repre)
                vals_list.append(value)
                repr_list.append(repre)

        if enc_type is not None:
            # Store already read encodings
            if reverted:
                encodings[enc_type] = {enc: int(val) for val, enc in zip(vals_list, repr_list)}
            else:
                encodings[enc_type] = {int(val): enc for val, enc in zip(vals_list, repr_list)}

    # If a unique encoding is specified, it's because both inputs and output have the same encodings
    if set(encodings.keys()) == set(
        [
            "both",
        ]
    ):
        encodings["input"] = encodings["both"]
        encodings["output"] = encodings["both"]

    return encodings


def replace_notech_primitives(synthed_filepath):
    """This function replace all verilog primitives in the synthesized design (for no_tech designs)."""
    primitives_template_dict = {
        "_AND_": "AND",
        "_NAND_": "NAND",
        "_OR_": "OR",
        "_NOR_": "NOR",
        "_NOT_": "NOT",
        "_XOR_": "XOR",
        "_XNOR_": "XNOR",
        "_ANDNOT_": "ANDNOT",
        "_ORNOT_": "ORNOT",
        "_MUX_": "MUX",
        "_DFF_PN0_": "DFF_PN0",
    }

    template_lines = synthed_filepath.read_text()
    # Clean the $ in front of gate names generated by yosys
    template = Template(template_lines.replace("\\$", "$"))
    template_substitute = template.substitute(primitives_template_dict)
    synthed_filepath.write_text(str(template_substitute))

    print(f"Gate names in syntrhesized netlist have been cleaned for simulation.")
    print(f"Synthesized netlist path is: {synthed_filepath}")

    return None


def generate_testbench(tb_template_filepath, design_filepath, synthed_filepath, out_dirpath, clean_synth_file=False):
    """
    This function reads the design, finds all wires, and substitute the template with the list of wires.
    All wires related to registers are removed from the wire list before substituting.
    """

    # Initialise files
    out_template_path = out_dirpath / "testbench.py"
    template_lines = tb_template_filepath.read_text()
    template = Template(template_lines)

    ### Testbench needs:
    # The wire list
    wire_list = get_wire_list(
        synthed_design_path=synthed_filepath, remove_dangling_wires=True, clean_output_pin_names=False
    )

    # The encoding dicts
    encodings_dict = extract_encodings(design_filepath)
    encodings_dict_reversed = extract_encodings(design_filepath, reverted=True)

    # Substitute in testbench template
    substitute_dict = {
        "wire_list": wire_list,
        "encodings_dict": str(encodings_dict),
        "encodings_dict_reversed": str(encodings_dict_reversed),
        # "validating_function": validating_function_txt,  (done manually for red zone adaptation)
        # "do_tests_list": tests_list_text,  (done manually for red zone adaptation)
    }
    template_substitute = template.substitute(substitute_dict)
    out_template_path.write_text(str(template_substitute))

    # If needed, clean the syntesized file so that it can be run easily
    if clean_synth_file:
        replace_notech_primitives(synthed_filepath=synthed_filepath)
    else:
        print(f"Synthesized netlist has NOT been cleaned for simulation.")
        print(f"To operate, add the CLI argument `--clean_synth` when launching this script.")

    return out_template_path


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(add_help=True)
    arg_parser.add_argument(
        "-t", "--tbtemp_filepath", type=Path, help="Relative or absolute path to the file where the testbench is stored"
    )
    arg_parser.add_argument(
        "-d", "--design_filepath", type=Path, help="Relative or absolute path to the file where the design is stored"
    )
    arg_parser.add_argument(
        "-s",
        "--synth_filepath",
        type=Path,
        help="Relative or absolute path to the file where the associated synthesized netlist is stored",
    )
    arg_parser.add_argument(
        "-o",
        "--output_dirpath",
        type=Path,
        help="Relative or absolute path to the diretory where the testbench will be stored",
    )
    arg_parser.add_argument(
        "--clean_synth",
        action="store_true",
        help="Whether or not to change gate names in the syntesized netlist. WARNING: operates inplace and will overwrite existing synthesized netlist.",
    )

    args = arg_parser.parse_args()

    for key, path in vars(args).items():
        if key in ["tbtemp_filepath", "design_filepath", "synth_filepath", "output_dirpath"]:
            if key == "output_dirpath" and not path.exists():
                path.mkdir(parents=True, exist_ok=True)
                print(f"Output directory `{path}` has been created.")
            assert path.exists(), f"The path {key} specified as `{path}` does not exist."

    testbench_path = generate_testbench(
        tb_template_filepath=args.tbtemp_filepath,
        design_filepath=args.design_filepath,
        synthed_filepath=args.synth_filepath,
        out_dirpath=args.output_dirpath,
        clean_synth_file=args.clean_synth,
    )

    print(f"Testbench has been generated and stored in file {testbench_path}.")
