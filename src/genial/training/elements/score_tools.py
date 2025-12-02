import numpy as np
import pandas as pd
from scipy.stats import gmean

from genial.utils.utils import extract_cont_str, convert_cont_str_to_np


class ScoreRescaler:
    @staticmethod
    def min_max_scaler(
        scores: np.ndarray | tuple[np.ndarray], return_args: bool = False, scaling_args: dict[str, float] = None
    ):
        # Deal with tuple of elements with recursion
        if isinstance(scores, tuple):
            scaled_scores = []
            for _scores in scores:
                _scores = ScoreRescaler.min_max_scaler(_scores, return_args=return_args, scaling_args=scaling_args)
                scaled_scores.append(_scores)
            return scaled_scores

        if scaling_args is not None:
            scale_min = scaling_args["min"]
            scale_max = scaling_args["max"]
        else:
            scale_min = np.amin(scores)
            scale_max = np.amax(scores)

        if return_args:
            scaling_args = {"min": scale_min, "max": scale_max}
            return ((scores - scale_min) / (scale_max - scale_min) - 0.5) * 2.0, scaling_args
        else:
            return ((scores - scale_min) / (scale_max - scale_min) - 0.5) * 2.0

    @staticmethod
    def standardize_scaler(
        scores: np.ndarray | tuple[np.ndarray], return_args: bool = False, scaling_args: dict[str, float] = None
    ):
        # Deal with tuple of elements with recursion
        if isinstance(scores, tuple):
            scaled_scores = []
            for _scores in scores:
                _scores = ScoreRescaler.standardize_scaler(_scores, return_args=return_args, scaling_args=scaling_args)
                scaled_scores.append(_scores)
            return scaled_scores

        if scaling_args is not None:
            mean = scaling_args["mean"]
            std = scaling_args["std"]
        else:
            mean = np.mean(scores)
            std = np.std(scores)

        if return_args:
            scaling_args = {"mean": mean, "std": std}
            return (scores - mean) / std, scaling_args
        else:
            return (scores - mean) / std

    @staticmethod
    def raw_scaler(scores: np.ndarray, return_args: bool = False, scaling_args: dict[str, float] = None, **kwargs):
        if isinstance(scores, tuple):
            scaled_scores = []
            for _scores in scores:
                _scores = ScoreRescaler.raw_scaler(_scores, return_args=return_args, scaling_args=scaling_args)
                scaled_scores.append(_scores)
            return scaled_scores

        if return_args:
            return scores, dict()
        else:
            return scores

    @staticmethod
    def min_max_descaler(scores: np.ndarray | tuple[np.ndarray], scaling_args: dict[str, float]):
        if isinstance(scores, tuple):
            raise NotImplementedError(
                f"{ScoreRescaler.__name__}.min_max_descaler() is not implemented for tuple of scores."
            )
        scale_min = scaling_args["min"]
        scale_max = scaling_args["max"]
        return (scores / 2.0 + 0.5) + (scale_max - scale_min) + scale_min

    @staticmethod
    def standardize_descaler(scores: np.ndarray | tuple[np.ndarray], scaling_args: dict[str, float]):
        if isinstance(scores, tuple):
            raise NotImplementedError(
                f"{ScoreRescaler.__name__}.standardize_descaler() is not implemented for tuple of scores."
            )
        mean = scaling_args["mean"]
        std = scaling_args["std"]
        return (scores * std) + mean

    @staticmethod
    def raw_descaler(scores: np.ndarray | tuple[np.ndarray], **kwrargs):
        if isinstance(scores, tuple):
            raise NotImplementedError(
                f"{ScoreRescaler.__name__}.raw_descaler() is not implemented for tuple of scores."
            )
        return scores


class ScoreTransformer:
    @staticmethod
    def trans_count(df: pd.DataFrame):  # Synth based
        if "nb_transistors" not in df.columns.tolist():
            raise ValueError(f"df of shape {df.shape} does not contain 'nb_transistors'. Available columns: {df.columns.tolist()}")
        return df["nb_transistors"].astype(int)

    @staticmethod
    def trans_depth(df: pd.DataFrame):  # Synth based
        return df["nb_transistors"].astype(int) / df["max_cell_depth"].astype(float)

    @staticmethod
    def swact_trans(df: pd.DataFrame):  # Swact based
        if "swact_weighted_total" in df.columns:
            swact_metric = "swact_weighted_total"
        elif "swact_weighted_average" in df.columns:
            swact_metric = "swact_weighted_average"
        return np.log(df[swact_metric].astype(float)) * df["nb_transistors"].astype(int)

    @staticmethod
    def swact(df: pd.DataFrame):  # Swact based
        if "swact_weighted_total" in df.columns:
            swact_metric = "swact_weighted_total"
        elif "swact_weighted_average" in df.columns:
            swact_metric = "swact_weighted_average"
        return np.log(df[swact_metric].astype(float))

    @staticmethod
    def swact_trans_depth(df: pd.DataFrame):  # Swact based
        if "swact_weighted_total" in df.columns:
            swact_metric = "swact_weighted_total"
        elif "swact_weighted_average" in df.columns:
            swact_metric = "swact_weighted_average"
        return np.log(df[swact_metric].astype(float)) * (
            df["nb_transistors"].astype(int) / df["max_cell_depth"].astype(float)
        )

    @staticmethod
    def power(df: pd.DataFrame):  # Power based
        return df["p_comb_dynamic"].astype(float)

    @staticmethod
    def composed_all_power(df: pd.DataFrame) -> tuple[pd.DataFrame]:  # Power based
        return df["p_comb_dynamic"].astype(float), df["tot_cell_area"].astype(float), df["max_delay_ps"].astype(float)

    @staticmethod
    def composed_power_area(df: pd.DataFrame) -> tuple[pd.DataFrame]:  # Power based
        return df["p_comb_dynamic"].astype(float), df["tot_cell_area"].astype(float)

    @staticmethod
    def composed_power(df: pd.DataFrame) -> tuple[pd.DataFrame]:  # Power based
        return (df["p_comb_dynamic"].astype(float),)

    @staticmethod
    def complexity(df: pd.DataFrame) -> tuple[pd.DataFrame]:  # CMPLX based
        return df["complexity_post_opt"].astype(int)


class MetricAlignmentHelper:
    rel_col_map = {
        "trans": ["nb_transistors"],
        "trans_depth": ["nb_transistors", "max_cell_depth"],
        "swact": ["swact_weighted_total", "swact_weighted_average"],
        "swact_trans": ["swact_weighted_total", "swact_weighted_average", "nb_transistors"],
        "swact_trans_depth": ["swact_weighted_total", "swact_weighted_average", "nb_transistors", "max_cell_depth"],
        "power": ["p_comb_dynamic"],
        "composed_all_power": ["p_comb_dynamic", "tot_cell_area", "max_delay_ps"],
        "composed_power_area": ["p_comb_dynamic", "tot_cell_area"],
        "composed_power": ["p_comb_dynamic"],
        "complexity": ["complexity_post_opt"],
    }

    @staticmethod
    def get_group_id(df: pd.DataFrame) -> pd.Series:
        """
        This method returns a group id for each design in the dataframe.
        """
        # Derive a boolean matrix representation of the encoding sequence
        encodings_input_np = (
            df["encodings_input"]
            .map(lambda x: extract_cont_str(x))
            .map(lambda x: convert_cont_str_to_np(x).reshape((16, 4)).T.astype(np.bool_))
        )
        # Derive the flipped representation of the encoding sequence
        encodings_input_np_flipped = encodings_input_np.map(lambda x: ~x)

        # Sort the columns and obtain by representation for unflipped and flipped version
        encodings_input_np_sorted = encodings_input_np.map(lambda x: (x[np.lexsort(x.T[::-1])]).tobytes())
        encodings_input_np_flipped_sorted = encodings_input_np_flipped.map(
            lambda x: (x[np.lexsort(x.T[::-1])]).tobytes()
        )

        return np.minimum(encodings_input_np_sorted, encodings_input_np_flipped_sorted)

    @staticmethod
    def merge_metric_according_to_invariance(df: pd.DataFrame, score_type: str) -> pd.DataFrame:
        """
        This method ensures that designs that are considered equivarient have the same target variable.

        First, an id for each design group is obtained. Then, the geometric mean of the elements in each
        group is assigned to each designs.
        """

        # Derive a boolean matrix representation of the encoding sequence
        encodings_input_np = (
            df["encodings_input"]
            .map(lambda x: extract_cont_str(x))
            .map(lambda x: convert_cont_str_to_np(x).reshape((16, 4)).T.astype(np.bool_))
        )
        # Derive the flipped representation of the encoding sequence
        encodings_input_np_flipped = encodings_input_np.map(lambda x: ~x)

        # Sort the columns and obtain by representation for unflipped and flipped version
        encodings_input_np_sorted = encodings_input_np.map(lambda x: (x[np.lexsort(x.T[::-1])]).tobytes())
        encodings_input_np_flipped_sorted = encodings_input_np_flipped.map(
            lambda x: (x[np.lexsort(x.T[::-1])]).tobytes()
        )

        # Obtain unique representation for the flipped and unflipped version
        df["encodings_input_group_id"] = np.minimum(encodings_input_np_sorted, encodings_input_np_flipped_sorted)

        # Find columns to adjust
        col_to_adjusts = [c for c in df.columns if c in MetricAlignmentHelper.rel_col_map[score_type]]

        # Assign geometric mean value for the group to each individual
        df_gr = df.groupby("encodings_input_group_id")
        for c in col_to_adjusts:
            # Get the geometric mean for the column
            temp_df_gr = df_gr[c].apply(gmean)
            # Create a map with the geometric mean for each group
            map_dict = dict(zip(temp_df_gr.index, temp_df_gr.values))
            # Assign the group value to the individual design
            df[c] = df["encodings_input_group_id"].map(map_dict)

        return df


class ScoreComputeHelper:
    scaler_map = {
        "minmax": ScoreRescaler.min_max_scaler,
        "standardize": ScoreRescaler.standardize_scaler,
        "raw": ScoreRescaler.raw_scaler,
    }

    descaler_map = {
        "minmax": ScoreRescaler.min_max_descaler,
        "standardize": ScoreRescaler.standardize_descaler,
        "raw": ScoreRescaler.raw_descaler,
    }

    transformer_map = {
        "trans": ScoreTransformer.trans_count,
        "trans_depth": ScoreTransformer.trans_depth,
        "swact": ScoreTransformer.swact,
        "swact_trans": ScoreTransformer.swact_trans,
        "swact_trans_depth": ScoreTransformer.swact_trans_depth,
        "power": ScoreTransformer.power,
        "composed_all_power": ScoreTransformer.composed_all_power,
        "composed_power_area": ScoreTransformer.composed_power_area,
        "composed_power": ScoreTransformer.composed_power,
        "complexity": ScoreTransformer.complexity,
    }

    @staticmethod
    def merge_data_df(right_df: pd.DataFrame, left_df: pd.DataFrame, suffix: str = "_synth"):
        """
        Merges synth and swact databases
        Also removes the duplicate design numbers datapoints
        """
        # Manually enforce to keep only the same design numbers
        common_design_numbers = set(right_df["design_number"]).intersection(set(left_df["design_number"]))
        _right_df = right_df[right_df["design_number"].isin(common_design_numbers)]
        _left_df = left_df[left_df["design_number"].isin(common_design_numbers)]
        merged_df = pd.merge(_left_df, _right_df, how="left", on="design_number", suffixes=("", suffix))
        return merged_df.loc[:, ~merged_df.columns.duplicated()]

    @staticmethod
    def compute_scores(
        score_type: str,
        score_rescale_mode: str,
        total_df: pd.DataFrame,
        return_args: bool = False,
        scaling_args: dict[str, float] = None,
    ):
        # Compute Raw Scores
        if score_type in ScoreComputeHelper.transformer_map:
            scores = ScoreComputeHelper.transformer_map[score_type](total_df)
        else:
            raise ValueError(f'Score type "{score_type}" not recognized!')

        # Rescales raw scores
        if score_rescale_mode in ScoreComputeHelper.scaler_map:
            return ScoreComputeHelper.scaler_map[score_rescale_mode](
                scores=scores,
                return_args=return_args,
                scaling_args=scaling_args,
            )
        else:
            raise ValueError(
                f"Arg `score_rescale_mode` should be among {ScoreComputeHelper.scaler_map.keys()}, "
                f"got {score_rescale_mode}"
            )
