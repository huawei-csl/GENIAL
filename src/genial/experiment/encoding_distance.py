from loguru import logger
import itertools

import torch
import numpy as np
from scipy.stats import kendalltau
import math

from genial.utils import utils

try:
    from Levenshtein import distance as levenshtein_distance
except Exception:
    logger.info(f"Tried to import Levenshtein distance but failed.")


def get_all_col_permuted_list(list1: list[str]):
    # Transpose lists
    list1_t = ["".join(column) for column in zip(*list1)]

    # Get all permutations
    all_list1 = list(
        itertools.permutations(
            list1_t,
        )
    )

    # Transpose back
    for idx in range(len(all_list1)):
        all_list1[idx] = ["".join(column) for column in zip(*all_list1[idx])]

    return all_list1


class EncodingsDistanceHelper:
    @staticmethod
    def _binary_to_set(binary_str: str) -> set[int]:
        return {i for i, bit in enumerate(binary_str) if bit == "1"}

    @staticmethod
    def _jaccard_distance(set1: set[int], set2: set[int]) -> float:
        intersection = set1.intersection(set2)
        union = set1.union(set2)
        return 1 - len(intersection) / len(union)

    @staticmethod
    def dictionary_jaccard_distance(dict1: dict[int, str], dict2: dict[int, str]) -> float:
        total_distance = 0
        for key in dict1.keys():
            set1 = EncodingsDistanceHelper._binary_to_set(dict1[key])
            set2 = EncodingsDistanceHelper._binary_to_set(dict2[key])
            total_distance += EncodingsDistanceHelper._jaccard_distance(set1, set2)
        return total_distance

    @staticmethod
    def dictionary_edit_distance(dict1: dict[int, str], dict2: dict[int, str]) -> float | None:
        try:
            total_distance = 0
            for key in dict1.keys():
                total_distance += levenshtein_distance(dict1[key], dict2[key])
            return total_distance
        except Exception:
            return None

    @staticmethod
    def list_edit_distance(list1: list[str], list2: list[str]) -> float | None:
        try:
            total_distance = 0
            for string1, string2 in zip(list1, list2):
                total_distance += levenshtein_distance(string1, string2)
            return total_distance
        except Exception:
            return None

    @staticmethod
    def flattened_list_edit_distance(list1: list[str], list2: list[str]) -> float | None:
        try:
            flattened_list1 = "".join(list1)
            flattened_list2 = "".join(list2)
            return levenshtein_distance(flattened_list1, flattened_list2)
        except Exception:
            return None

    @staticmethod
    def list_permutation_distance(list1: list[str], list2: list[str]):
        # Step 1: Create a mapping from elements to their positions in list1
        position_map = {value: index for index, value in enumerate(list1)}

        # Step 2: Mark visited elements
        visited = [False] * len(list1)
        total_distance = 0

        # Step 3: Count cycles and calculate the distance
        for i in range(len(list1)):
            if not visited[i]:
                cycle_length = 0
                j = i

                while not visited[j]:
                    visited[j] = True

                    # Deal with the case where list2[j] is not in position_map.keys()
                    if list2[j] not in position_map.keys():
                        # If list2[j] is not in position_map.keys(), it means it's a new element not present in list1.
                        # In this case, we can't calculate the distance for this element, so we add the worst case distance (len(list1) - 1)
                        total_distance += len(list1) - 1
                        break

                    j = position_map[list2[j]]
                    cycle_length += 1

                # Step 4: Add to total distance (k - 1 for each cycle of length k)
                if cycle_length > 0:
                    total_distance += cycle_length - 1

        return total_distance

    @staticmethod
    def convert_to_multiclass(binary_tensor: torch.Tensor, nb_classes: int = 2) -> torch.Tensor:
        """
        Convert a binary tensor to a multiclass tensor.
        Args:
            binary_tensor (torch.Tensor): Binary tensor to convert.
            nb_classes (int): Number of classes.
        Returns:
            torch.Tensor: Multiclass tensor.
        """
        # Create a tensor to store the multiclass representation
        multiclass_tensor = torch.zeros_like(binary_tensor, dtype=torch.int8)
        multiclass_tensor = torch.stack([multiclass_tensor] * nb_classes, dim=0)

        # Duplicate the binary tensor along the channel dimension
        for i in range(nb_classes):
            multiclass_tensor[i, ...][binary_tensor == i] = 1

        return multiclass_tensor

    @staticmethod
    def list_dice_loss(list1: list[str], list2: list[str]):
        from genial.training.elements.metrics.metrics import DiceLoss

        array1 = torch.tensor(utils.from_binstr_list_to_int_array(list1))
        array2 = torch.tensor(utils.from_binstr_list_to_int_array(list2))

        array1 = EncodingsDistanceHelper.convert_to_multiclass(array1)
        array2 = EncodingsDistanceHelper.convert_to_multiclass(array2)

        return DiceLoss().dice_loss_func(array2, array1).item()

    @staticmethod
    def list_column_permutation_distance(list1: list[str], list2: list[str], weight: bool = False):
        # Transpose the lists
        transposed_list1 = ["".join(column) for column in zip(*list1)]
        transposed_list2 = ["".join(column) for column in zip(*list2)]

        if weight:
            c_b_r_ratio = len(list1) * 1.0 / len(transposed_list2)
        else:
            c_b_r_ratio = 1

        return EncodingsDistanceHelper.list_permutation_distance(transposed_list1, transposed_list2) * c_b_r_ratio

    @staticmethod
    def list_kendall_tau_distance(list1: list[str], list2: list[str]):
        # Create a ranking for each list based on the strings
        rank1 = {s: i for i, s in enumerate(list1)}
        rank2 = {s: i for i, s in enumerate(list2)}

        # Find common elements in both lists
        common_elements = set(list1).intersection(set(list2))

        # If there are no common elements, the distance is undefined or could be considered as 1 (max distance)
        if not common_elements:
            return 1.0

        # Convert common elements to rankings from both lists
        ranked_list1 = [rank1[s] for s in common_elements]
        ranked_list2 = [rank2[s] for s in common_elements]

        # Calculate Kendall Tau
        return 1 - kendalltau(ranked_list1, ranked_list2).correlation

    @staticmethod
    def list_column_kendall_tau_distance(list1: list[str], list2: list[str], weight: bool = False):
        # Transpose the lists
        transposed_list1 = ["".join(column) for column in zip(*list1)]
        transposed_list2 = ["".join(column) for column in zip(*list2)]

        return EncodingsDistanceHelper.list_kendall_tau_distance(transposed_list1, transposed_list2)

    @staticmethod
    def hamming_distance(s1, s2):
        """Calculate the Hamming distance between two binary strings of equal length."""
        return sum(c1 != c2 for c1, c2 in zip(s1, s2))

    @staticmethod
    def list_hamming_distance(list1: list[str], list2: list[str]):
        """Calculate the total Hamming distance between two lists of binary strings."""
        binary_array1 = utils.from_binstr_list_to_int_array(list1)
        binary_array2 = utils.from_binstr_list_to_int_array(list2)
        return np.sum(np.abs(binary_array1 - binary_array2))

    @staticmethod
    def list_smart_kendall_tau_distance(list1: list[str], list2: list[str]):
        """
        This function measure the smart Kendal Tau distance
        1) Reorder lists if some of their columns can be matched
        2) Measure the 1D row-wise Kendall Tau distance

        """
        _list1, _list2 = EncodingsDistanceHelper.align_columns_if_match(list1, list2)

        # Compute the distance
        distance_after = EncodingsDistanceHelper.list_kendall_tau_distance(_list1, _list2)

        return distance_after

    @staticmethod
    def list_smart_hamming_distance(list1: list[str], list2: list[str]):
        """
        This function measure the smart Hamming distance
        1) Reorder lists if some of their columns can be matched
        2) Measure the 1D row-wise Hamming distance

        """
        _list1, _list2 = EncodingsDistanceHelper.align_columns_if_match(list1, list2)

        # Compute the distance
        distance_after = EncodingsDistanceHelper.list_hamming_distance(_list1, _list2)

        return distance_after

    @staticmethod
    def minimum_swaps_to_convert(list1: list[str], list2: list[str]):
        """
        Compute the minimum number of swaps required to convert list list1 into list list2.
        Both list1 and list2 contain the same elements.
        """
        # Create a value-to-index map for list2 to know where each element of list1 should go
        index_map_B = {value: idx for idx, value in enumerate(list2)}

        visited = [False] * len(list1)
        swaps = 0

        for i in range(len(list1)):
            # If already visited or is already in correct place, skip
            if visited[i] or index_map_B[list1[i]] == i:
                visited[i] = True
                continue

            # Compute the size of the cycle
            cycle_size = 0
            current_index = i

            while not visited[current_index]:
                visited[current_index] = True
                # Move to the index where the current element should be
                current_index = index_map_B[list1[current_index]]
                cycle_size += 1

            # If there is a cycle of length k, we need (k-1) swaps
            if cycle_size > 0:
                swaps += cycle_size - 1

        return swaps

    @staticmethod
    def rotate_list(_list: list[str], k: int):
        return _list[k:] + _list[:k]

    @staticmethod
    def cyclic_kendall_tau_distance(list1: list[str], list2: list[str], nb_cycles: int = 1):
        min_distance = float("inf")
        if nb_cycles > len(list1):
            range_iterator = range(len(list1))
        else:
            range_iterator = range(-nb_cycles, nb_cycles + 1, 1)
        for k in range_iterator:
            list2_rot = EncodingsDistanceHelper.rotate_list(list2, k)
            distance = EncodingsDistanceHelper.list_kendall_tau_distance(list1, list2_rot)

            if distance < min_distance:
                min_distance = distance

        return min_distance

    @staticmethod
    def cyclic_dice_loss_distance(list1: list[str], list2: list[str], nb_cycles: int = 1):
        min_distance = float("inf")
        if nb_cycles > len(list1) / 2.0:
            range_iterator = range(len(list1))
        else:
            range_iterator = range(-nb_cycles, nb_cycles + 1, 1)
        for k in range_iterator:
            list2_rot = EncodingsDistanceHelper.rotate_list(list2, k)
            distance = EncodingsDistanceHelper.list_dice_loss(list1, list2_rot)

            if distance < min_distance:
                min_distance = distance

        return min_distance

    @staticmethod
    def list_smart_composed_distance(
        target_list: list[str], evaluated_list: list[str], return_info: bool = False, nb_cycles: int = None
    ) -> tuple[float, bool]:
        """
        This function measure the smart composed distance
        Args:
            target_list (list[str]): The target list of binary strings
            evaluated_list (list[str]): The evaluated list of binary strings
            return_info (bool, optional): If True, return the distance, the negated flag and the aligned list. Defaults to False.
        Returns:
            tuple[float, bool]: The distance and the negated flag
        """

        # Get aligned and aligned negated list to be tested
        negated_eval_list = utils.negate_list_binstr(evaluated_list)
        target_list, aligned_eval_list = EncodingsDistanceHelper.align_columns_if_match(target_list, evaluated_list)
        target_list, aligned_negated_eval_list = EncodingsDistanceHelper.align_columns_if_match(
            target_list, negated_eval_list
        )

        # Measure hamming distance on both lists
        default_hamming_distance = EncodingsDistanceHelper.list_hamming_distance(target_list, aligned_eval_list)
        negated_hamming_distance = EncodingsDistanceHelper.list_hamming_distance(target_list, aligned_negated_eval_list)

        # Keep the list with minimum distance
        is_negated = False
        if default_hamming_distance <= negated_hamming_distance:
            eval_list = aligned_eval_list
        else:
            eval_list = aligned_negated_eval_list
            is_negated = True

        # Measure the distance for all permutations of the columns
        # Measure cyclic distance
        if nb_cycles is None:
            nb_cycles = math.ceil(len(target_list) / 2.0)

        # final_distance = EncodingsDistanceHelper.cyclic_kendall_tau_distance(target_list, eval_list, n_cycles=nb_cycles)
        final_distance = EncodingsDistanceHelper.cyclic_dice_loss_distance(target_list, eval_list, nb_cycles=nb_cycles)

        # if final_distance > 1.70:
        # print("HERE")

        if return_info:
            return final_distance, is_negated
        else:
            return final_distance

    @staticmethod
    def list_smarter_composed_distance(
        target_list: list[str], evaluated_list: list[str], return_info: bool = False, nb_cycles: int = None
    ) -> tuple[float, bool]:
        """
        This function measure the smart composed distance
        Args:
            target_list (list[str]): The target list of binary strings
            evaluated_list (list[str]): The evaluated list of binary strings
            return_info (bool, optional): If True, return the distance, the negated flag and the aligned list. Defaults to False.
        Returns:
            tuple[float, bool]: The distance and the negated flag
        """

        # Get aligned and aligned negated list to be tested
        negated_eval_list = utils.negate_list_binstr(evaluated_list)

        distances = []
        distance_min = float("inf")
        is_negated = False
        for idx, eval_list in enumerate([evaluated_list, negated_eval_list]):
            eval_lists = get_all_col_permuted_list(eval_list)
            for _eval_list in eval_lists:
                distances.append(
                    EncodingsDistanceHelper.cyclic_dice_loss_distance(target_list, _eval_list, nb_cycles=nb_cycles)
                )

                if distance_min > distances[-1]:
                    distance_min = distances[-1]
                    if idx == 1:
                        is_negated = True

        if return_info:
            return distance_min, is_negated
        else:
            return distance_min

    @staticmethod
    def list_smarter_kendall_tau_distance(
        target_list: list[str], evaluated_list: list[str], return_info: bool = False, nb_cycles: int = None
    ) -> tuple[float, bool]:
        """
        This function measure the smart composed distance
        Args:
            target_list (list[str]): The target list of binary strings
            evaluated_list (list[str]): The evaluated list of binary strings
            return_info (bool, optional): If True, return the distance, the negated flag and the aligned list. Defaults to False.
        Returns:
            tuple[float, bool]: The distance and the negated flag
        """

        # Get aligned and aligned negated list to be tested
        negated_eval_list = utils.negate_list_binstr(evaluated_list)

        distances = []
        distance_min = float("inf")
        is_negated = False
        for idx, eval_list in enumerate([evaluated_list, negated_eval_list]):
            eval_lists = get_all_col_permuted_list(eval_list)
            for _eval_list in eval_lists:
                distances.append(
                    EncodingsDistanceHelper.cyclic_kendall_tau_distance(target_list, _eval_list, nb_cycles=nb_cycles)
                )

                if distance_min > distances[-1]:
                    distance_min = distances[-1]
                    if idx == 1:
                        is_negated = True

        if return_info:
            return distance_min, is_negated
        else:
            return distance_min

    @staticmethod
    def align_columns_if_match(list1: list[str], list2: list[str]) -> list[str]:
        """
        This function aligns list2 on list1
        Warning: list1 does not change, only list2 changes
        """
        # Transpose the lists
        transposed_list1 = ["".join(column) for column in zip(*list1)]
        transposed_list2 = ["".join(column) for column in zip(*list2)]

        # Look for shared columns
        transposed_set1 = set(transposed_list1)
        transposed_set2 = set(transposed_list2)
        shared_columns = []
        for col in transposed_set1:
            if col in transposed_set2:
                shared_columns.append(col)

        if len(shared_columns) != 0:
            # Re-organize the lists so that the shared columns are at the same index in both lists
            for col in shared_columns:
                idx1 = transposed_list1.index(col)
                idx2 = transposed_list2.index(col)
                if idx1 != idx2:
                    tmp_col2 = transposed_list2[idx1]
                    transposed_list2[idx1] = transposed_list1[idx1]
                    # print(f"Replaced {transposed_list2[idx2]}")
                    transposed_list2[idx2] = tmp_col2
                    # print(f"by {transposed_list2[idx2]}")

            # Transpose back the lists
            # _list1 = [''.join(column) for column in zip(*transposed_list1)]
            _list2 = ["".join(column) for column in zip(*transposed_list2)]

            return list1, _list2

        else:
            return list1, list2
