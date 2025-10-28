import torch
from genial.experiment.task_generator import DesignGenerator
from genial.utils.utils import from_int_array_to_binstr_array


def convert_proto_to_encoding_dictionary(dir_config, fixed_prototypes_batched):
    if fixed_prototypes_batched is None:
        pass

    # Convert back the fixed prototype to a dictionnary
    if isinstance(fixed_prototypes_batched, list):
        _fixed_prototypes_batched = []
        for fixed_prototype in fixed_prototypes_batched:
            if fixed_prototype is not None:
                _fixed_prototypes_batched.append(fixed_prototype)
        _fixed_prototypes_batched = torch.stack(_fixed_prototypes_batched)
    else:
        _fixed_prototypes_batched = fixed_prototypes_batched

    assert (
        _fixed_prototypes_batched.ndim == 3
    )  # We expect the prototpye to have the shape (batch_size, nb_values, bitwidth)

    proto_encoding_dicts_list = []
    for fixed_prototype in _fixed_prototypes_batched:
        fixed_prototype_arr = fixed_prototype.type(torch.int32).numpy()
        fixed_prototype_strlist = from_int_array_to_binstr_array(fixed_prototype_arr, return_as_list=True)

        values_dict = DesignGenerator.get_all_sorted_unique_values(
            in_enc_type=dir_config.exp_config["input_encoding_type"],
            in_bitwidth=int(dir_config.exp_config["input_bitwidth"]),
            design_type=dir_config.exp_config["design_type"],
        )
        out_representations = DesignGenerator.get_binary_representations(
            values_sorted=values_dict["output"],
            enc_type=dir_config.exp_config["output_encoding_type"],
            bitwidth=dir_config.exp_config["output_bitwidth"],
            port="output",
        )
        protoype_as_dict = {
            "in_enc_dict": {int(val): repr for val, repr in zip(values_dict["input"], fixed_prototype_strlist)},
            "out_enc_dict": {int(val): repr for val, repr in zip(values_dict["output"], out_representations)},
        }
        proto_encoding_dicts_list.append(protoype_as_dict)

    return proto_encoding_dicts_list
