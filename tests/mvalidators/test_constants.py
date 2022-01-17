from strictyaml import load, YAML
from mvalidators.constants import Constants

def test_open():
    constants = Constants()
    constants_yml = YAML(constants.__dict__)
    assert constants_yml is not None