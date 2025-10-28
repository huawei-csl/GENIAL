from genial.globals import global_vars

if global_vars["flowy_available"]:
    from flowy.utils.design_generation.hdl_generator import HdlGenerator as FlowyHdlGenerator
else:
    FlowyHdlGenerator = object

import pandas as pd


class AdaptedFlowyHdlGenerator(FlowyHdlGenerator):
    def __init__(
        self,
        truth_table: pd.DataFrame,
    ):
        _truth_table = self.convert_to_dict_of_binstr(truth_table)

        try:
            super().__init__(_truth_table)
        except TypeError:
            # If flowy is not available
            super().__init__()

    def convert_to_dict_of_binstr(self, truth_table: pd.DataFrame):
        if len(truth_table.columns) == 4:
            nb_inputs = 1
        elif len(truth_table.columns) == 6:
            nb_inputs = 2
        else:
            raise ValueError("Truth table must have 4 or 6 columns")

        lut_dict = {}
        for idx, row in truth_table.iterrows():
            if nb_inputs == 1:
                _key = str(row["input_a_rep"])
                _val = str(row["output_rep"])
            elif nb_inputs == 2:
                _key = (str(row["input_a_rep"]), str(row["input_b_rep"]))
                _val = str(row["output_rep"])
            else:
                raise ValueError("Truth table must have 4 or 6 columns")
            lut_dict[_key] = _val

        return lut_dict

    def main_genial(self, do_optimize: bool):
        try:
            complexity_pre_opt = self.get_complexity(self.truth_table_dict)
            truth_table_dict_opt = self.optimize_lookup_table(self.truth_table_dict)
            complexity_post_opt = self.get_complexity(truth_table_dict_opt)

            if do_optimize:
                self.truth_table_dict = truth_table_dict_opt

            # complexity_pre_opt = self.get_complexity()
            # if do_optimize:
            #     self.optimize_lookup_table()
            # complexity_post_opt = self.get_complexity()

            if self.truth_table_has_dontcares():
                hdl_lookup = self.get_lookup_code_minterm()
            else:
                hdl_lookup = self.get_lookup_code()

            # Add complexity line to generated hdl
            complexity_info_dict = {
                "complexity_pre_opt": complexity_pre_opt,
                "complexity_post_opt": complexity_post_opt,
                "was_optimized": do_optimize,
            }

            return hdl_lookup, complexity_info_dict

        except AttributeError:
            complexity_info_dict = {
                "complexity_pre_opt": 7777,
                "complexity_post_opt": 7777,
                "was_optimized": False,
            }
            return None, complexity_info_dict
