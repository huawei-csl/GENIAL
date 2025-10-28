# SPDX-License-Identifier: BSD-3-Clause Clear
# Copyright (c) 2024-2025 HUAWEI.
# All rights reserved.
#
# This software is released under the BSD 3-Clause Clear License.
# See the LICENSE file in the project root for full license information (https://github.com/huawei-csl/GENIAL).

# Code adapted from
# https://github.com/elizabethjoshi/BinaryCalculations
# License: GPL-3.0

from loguru import logger


def BinaryAdd(x, y):  # function to carry out binary addition and returns the result as a string
    maxlen = max(len(x), len(y))

    # Normalize lengths
    x = x.zfill(maxlen)
    y = y.zfill(maxlen)
    result = ""
    carry = 0
    for i in range(maxlen - 1, -1, -1):  # reverse order range, decreasing by 1 in every iteration
        r = carry
        r += 1 if x[i] == "1" else 0
        r += 1 if y[i] == "1" else 0
        result = ("1" if r % 2 == 1 else "0") + result
        carry = 0 if r < 2 else 1
    if carry != 0:
        result = "1" + result
    result.zfill(maxlen)
    return result[-16:]


def min_value_tc(bitwidth: int) -> int:
    """Return the minimum value (included) that can be represented on <bitwidth> bit using two's complement representation."""
    return -(2 ** (bitwidth - 1))


def max_value_tc(bitwidth: int) -> int:
    """Return the maximum value (included) that can be represented on <bitwidth> bit using two's complement representation."""
    return (2 ** (bitwidth - 1)) - 1


def twos_complement(value: int, bitwidth: int, **kwargs) -> str:
    """Returns the 2s complement representation of input value binary input. Bitwidth is the total available bit width, sign bit included."""

    binstr = bin(value).replace("0b", "").replace("-", "").zfill(bitwidth)
    if value >= 0:
        if value > max_value_tc(bitwidth):
            logger.warning(
                f"The two's complement binary representation of {value} cannot exist on {bitwidth} bits. Maximum value (included) is: {max_value_tc(bitwidth)}. Value will be skipped."
            )
            return None
        else:
            return binstr
    else:
        if value < min_value_tc(bitwidth):
            logger.warning(
                f"The two's complement binary representation of {value} cannot exist on {bitwidth} bits. Minimum value (included) is: {min_value_tc(bitwidth)}. Value will be skipped."
            )
            return None
        else:
            li = list(binstr)
            for i in range(len(li)):
                li[i] = "0" if li[i] == "1" else "1"
            return BinaryAdd("".join(li), "1")


def min_value_uint(bitwidth: int) -> int:
    """Return the minimum value (included) that can be represented on <bitwidth> bit using unsigned interger representation."""
    return 0


def max_value_uint(bitwidth: int) -> int:
    """Return the maximum value (included) that can be represented on <bitwidth> bit using unsigned interger representation."""
    return (2 ** (bitwidth)) - 1


def unsigned_integer(value: int, bitwidth: int, **kwargs) -> str:
    """Returns the 2s complement representation of input value binary input. Bitwidth is the total available bit width, sign bit included."""

    binstr = bin(value).replace("0b", "").replace("-", "").zfill(bitwidth)

    if value > max_value_uint(bitwidth):
        logger.warning(
            f"The unigned binary representation of {value} cannot exist on {bitwidth} bits. Maximum value (included) is: {max_value_uint(bitwidth)}. Value will be skipped."
        )
        return None
    elif value < min_value_uint(bitwidth):
        logger.warning(
            f"The unigned binary representation of {value} cannot exist on {bitwidth} bits. Minimum value (included) is: {min_value_uint(bitwidth)}. Value will be skipped."
        )
        return None
    else:
        return binstr


def RightShift(a):  # function to carry out right shift
    a = list(a)
    for i in range(len(a) - 1, 0, -1):
        a[i] = a[i - 1]
    return "".join(a)


def LeftShift(a):  # function to carry out left shift
    a = list(a)
    return "".join(a[-(len(a) - 1) :])
