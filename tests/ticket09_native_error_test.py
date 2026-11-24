"""Flat-native error diagnostics only; never independent physical evidence."""
import re

import pytest

from pyhygrothermfem_fairyfly import _native

from ticket09_acceptance_cases import INVALID_MATERIAL_VALUES


FIELD_POSITION = {
    'BulkDensity': (0, r'densit|BulkDensity'),
    'Porosity': (1, r'porosity|Porosity'),
    'SpecificHeatCapacityDry': (
        2, r'heat.capacit|heat_capacity|SpecificHeatCapacityDry'),
    'WaterVaporDiffusionResistanceFactor': (
        3, r'diffusion.resistance|diffusion_resistance|'
           r'WaterVaporDiffusionResistanceFactor'),
}


@pytest.mark.parametrize('field,value', INVALID_MATERIAL_VALUES)
def test_native_material_error_identifies_received_value(field, value):
    """A flat-entry rejection identifies its material index and bad scalar.

    Invalid material data is rejected before timestep advancement. This
    tests the pre-agreed flat boundary's error format, not solver physics.
    """
    nodes = [(1, 0.0, 0.0), (2, 2.0, 0.0),
             (3, 2.0, 1.0), (4, 0.0, 1.0)]
    quads = [(1, 2, 3, 4)]
    materials = [(1.0, 0.9, 0.9, '', '', '', False, 1.0, 101325.0)]
    properties = [600.0, 0.2, 850.0, 50.0,
                  [(0.0, 0.0), (0.5, 75.0), (1.0, 150.0)],
                  [], [], 0.0, [], 0.0]
    position, field_pattern = FIELD_POSITION[field]
    properties[position] = value
    with pytest.raises(ValueError) as caught:
        _native.solve_hygrothermal(
            nodes, quads, materials, [tuple(properties)],
            (True, True, True, True, True, False),
            convection=[(2, 3, (10.0,), (20.0,))],
            time_step=1.0, num_steps=1)
    message = str(caught.value)
    assert re.search(field_pattern, message, re.IGNORECASE)
    assert repr(value) in message
    assert 'material' in message.lower()
    assert re.search(r'\b[01]\b', message)
