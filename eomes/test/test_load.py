"""Test eomes.load."""

import numpy as np
from eomes.load import parse_inputfile, ParsedParams, check_inputs
from .tools import find_datafiles

import pytest


class MockParsedParams(ParsedParams):
    """Mocking load module class ParsedParams
    """

    def __init__(self):
        pass


def content_dict(filename):
    """A tool function to parse content from a file.

    Args:
        filename (str): Input file.

    Returns:
        dict: Parsed input parameters.
    """
    with open(filename, "r") as ifile:
        content = ifile.read()
    # Make list of lines in ifile skipping white lines.
    # Based on stackoverflow post:
    # how-to-delete-all-blank-lines-in-the-file-with-the-help-of-python
    lines = [line for line in content.split("\n") if line.strip()]
    kwargs = {l.split(":")[0].strip(): l.split(":")[1].strip() for l in lines}
    return kwargs


def parsed_examples():
    """Define test cases to test the load module.

    Yields:
        tuple: Input parameters for test_check_inputs
    """
    cases = [
        (
            "test_input1.in",
            [
                (2, 2),
                "eomes/test/data/be_sto3g_oneint_spino.npy",
                "eomes/test/data/be_sto3g_twoint_spino.npy",
                "eomes/test/data/be_sto3g_onedm_spino.npy",
                "eomes/test/data/be_sto3g_twodm_spino.npy",
                "symmetric",
                "ip",
                False,
                None,
                1.0e-7,
            ],
        ),
        (
            "test_input2.in",
            [
                4,
                "eomes/test/data/be_sto3g_oneint_spino.npy",
                "eomes/test/data/be_sto3g_twoint_spino.npy",
                "eomes/test/data/be_sto3g_onedm_spino.npy",
                "eomes/test/data/be_sto3g_twodm_spino.npy",
                "asymmetric",
                "ea",
                True,
                1,
                1.0e-6,
            ],
        ),
    ]
    for case in cases:
        yield case


@pytest.mark.parametrize(
    "filename, expected", parsed_examples(),
)
def test_parse_inputfile(filename, expected):
    inputfile = find_datafiles(filename)
    test_params = parse_inputfile(inputfile)
    assert np.allclose(test_params.nparts, expected[0])
    assert test_params.oneint_file == expected[1]
    assert test_params.twoint_file == expected[2]
    assert test_params.dm1_file == expected[3]
    assert test_params.dm2_file == expected[4]
    assert test_params.orthog == expected[5]
    assert test_params.eom == expected[6]
    # Optional inputs
    assert test_params.get_tdm is expected[7]
    assert test_params.roots is expected[8]
    assert np.allclose(test_params.tol, expected[9])


def test_parsedparams_errors():
    # content = {
    #     "nelec": "(2, 2)",
    #     "tol": "1e-6",
    #     "oneint_file": "be_sto3g_oneint_spino.npy",
    #     "twoint_file": "be_sto3g_twoint_spino.npy",
    #     "dm1_file": "be_sto3g_onedm_spino.npy",
    #     "dm2_file": "be_sto3g_twodm_spino.npy",
    #     "orthog": "symmetric",
    #     "eom": "ip",
    #     "roots": "1",
    #     "get_tdm": "False",
    # }
    inputfile = find_datafiles("test_input2.in")
    content = content_dict(inputfile)
    test_params = ParsedParams(content)

    # Check optional inputs
    content["get_tdm"] = "False"
    test_params = ParsedParams(content)
    assert test_params.get_tdm is False

    # Check errors
    content["nelec"] = ""
    with pytest.raises(TypeError):
        ParsedParams(content)

    content["nelec"] = "2"
    test_params = ParsedParams(content)
    assert np.allclose(test_params.nparts, 2)

    content["get_tdm"] = "Yes"
    with pytest.raises(ValueError):
        ParsedParams(content)


def example_params():
    """Define test cases to test the load module.

    Yields:
        tuple: Input parameters for test_check_inputs
    """
    vals = [
        (2, 2),
        0.1,
        None,
        False,
        find_datafiles("be_sto3g_oneint_spino.npy"),
        find_datafiles("be_sto3g_twoint_spino.npy"),
        find_datafiles("be_sto3g_onedm_spino.npy"),
        find_datafiles("be_sto3g_twodm_spino.npy"),
        "symmetric",
        "ip",
    ]
    wrong_vals = [
        vals[1],
        vals[0],
        vals[1],
        vals[-1],
        "temp.npy",
        "temp.npy",
        "temp.npy",
        "temp.npy",
        "ip",
        "symmetric",
    ]
    cases = []
    for i in range(10):
        print(i)
        temp = vals[:]
        temp[i] = wrong_vals[i]
        cases.append(temp)

    tests = zip(cases, [TypeError] * 4 + [FileNotFoundError] * 4 + [ValueError] * 2)
    for case, error in tests:
        yield (case, error)


@pytest.mark.parametrize(
    "test, error", example_params(),
)
def test_wrong_inputs(test, error):
    mock_params = MockParsedParams()
    mock_params.nparts = test[0]
    mock_params.tol = test[1]
    mock_params.roots = test[2]
    mock_params.get_tdm = test[3]
    mock_params.oneint_file = test[4]
    mock_params.twoint_file = test[5]
    mock_params.dm1_file = test[6]
    mock_params.dm2_file = test[7]
    mock_params.orthog = test[8]
    mock_params.eom = test[9]

    with pytest.raises(error):
        check_inputs(mock_params)


def test_check_inputs_parsedparams():
    class Temp:
        def __init__(self):
            pass

    mock_params = Temp()

    with pytest.raises(TypeError):
        check_inputs(mock_params)
