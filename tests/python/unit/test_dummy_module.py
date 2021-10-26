# -*- coding: utf-8 -*-
# This is a test file intended to be used with pytest
# pytest automatically runs all the function starting with "test_"
# see https://docs.pytest.org for more information

from lib import dummy_function


def test_dummy_function():
    dummy_results = dummy_function()
    assert dummy_results == "foo"
