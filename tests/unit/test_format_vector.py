"""Unit tests for format_vector function."""

import pytest
import math
from app import format_vector


def test_format_vector_basic():
    vec = [0.1, -0.2, 0.0]
    assert format_vector(vec) == "[0.10000000,-0.20000000,0.00000000]"


def test_format_vector_nan_inf():
    vec = [float('nan'), float('inf'), float('-inf'), 0.5]
    # NaN/Inf should be replaced with 0.0
    result = format_vector(vec)
    assert result == "[0.00000000,0.00000000,0.00000000,0.50000000]"


def test_format_vector_empty():
    assert format_vector([]) == "[]"


def test_format_vector_precision():
    vec = [0.123456789, -0.987654321]
    result = format_vector(vec)
    assert result == "[0.12345679,-0.98765432]"