"""Public rejection and atomic-publication checks for declared wet limits."""
import xml.etree.ElementTree as ET
import zipfile

import pytest

from pyhygrothermfem_fairyfly import ContractError, UnsupportedError, inspect, run_thmz, solve

from ticket09_acceptance_cases import wet_slab


REJECTIONS = (
    ('steady-moisture', 'THMZ-E5003'),
    ('moisture-only', 'THMZ-E5003'),
    ('mixed-lambda-temperature', 'THMZ-E5010'),
    ('mixed-lambda-moisture', 'THMZ-E5010'),
    ('redistribution', 'THMZ-E5010'),
    ('mu-water', 'THMZ-E5010'),
    ('enthalpy-temperature', 'THMZ-E5010'),
    ('missing-porosity', 'THMZ-E2003'),
    ('missing-sorption', 'THMZ-E2003'),
    ('missing-liquid-transport', 'THMZ-E2004'),
    ('missing-capillary-transport', 'THMZ-E2004'),
    ('liquid-saturation-mismatch', 'THMZ-E2005'),
    ('missing-lambda-tables', 'THMZ-E2004'),
    ('lambda-reference-mismatch', 'THMZ-E2005'),
    ('lambda-water-coverage', 'THMZ-E2005'),
    ('neumann-without-convection', 'THMZ-E5016'),
)


def _table(parent, name, values):
    table = ET.SubElement(parent, name)
    for x, y in values:
        row = ET.SubElement(table, 'TableValue')
        ET.SubElement(row, 'x').text = str(x)
        ET.SubElement(row, 'y').text = str(y)


def _rejected_input(path, case):
    """Mutate only documented XML fields; no production parsing is reused."""
    wet_slab(path)
    with zipfile.ZipFile(path) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    model = ET.fromstring(members['Model.xml'])
    options = model.find('Properties/CalculationOptions')
    flags = options.find('ModelingOptions')
    material = ET.fromstring(members['Materials.xml'])
    hygro = material.find('Material/Solid/HygroThermal')
    types = ET.fromstring(members['TransientTypeBC.xml'])
    if case == 'steady-moisture':
        options.find('CalculationMode').text = 'Steady-State'
    elif case == 'moisture-only':
        options.find('SimulateThermal').text = 'false'
    elif case.startswith('mixed-lambda-'):
        suffix = 'Temperature' if case.endswith('temperature') else 'Moisture'
        flags.find('ExcludeThermalConductivity' + suffix + 'Dependent').text = 'false'
    elif case in ('redistribution', 'mu-water', 'enthalpy-temperature'):
        tag = {'redistribution': 'LiquidTransportationCoefficientRedistribution',
               'mu-water': 'WaterVaporDiffusionResistanceFactorMoistureDependent',
               'enthalpy-temperature': 'EnthalpyTemperatureDependent'}[case]
        _table(hygro, tag, ((0, 1), (150, 1)))
    elif case in ('missing-porosity', 'missing-sorption'):
        tag = 'Porosity' if case.endswith('porosity') else 'MoistureStorageFunction'
        hygro.remove(hygro.find(tag))
    elif case in ('missing-liquid-transport', 'missing-capillary-transport',
                  'liquid-saturation-mismatch'):
        flag = 'ExcludeCapillaryConduction' if case == 'missing-capillary-transport' \
            else 'ExcludeWaterLiquidTransportation'
        flags.find(flag).text = 'false'
        if case == 'liquid-saturation-mismatch':
            _table(hygro, 'LiquidTransportationCoefficientSuction',
                   ((0, 1e-8), (100, 1e-8)))
    elif case.startswith('lambda-') or case == 'missing-lambda-tables':
        for name in ('Moisture', 'Temperature'):
            flags.find('ExcludeThermalConductivity' + name + 'Dependent').text = 'false'
        if case != 'missing-lambda-tables':
            upper = 100 if case == 'lambda-water-coverage' else 150
            _table(hygro, 'ThermalConductivityMoistureDependent',
                   ((0, 1), (upper, 1.5)))
            constant = 2 if case == 'lambda-reference-mismatch' else 1
            _table(hygro, 'ThermalConductivityTemperatureDependent',
                   ((0, constant), (60, constant)))
    elif case == 'neumann-without-convection':
        record = types.find('BoundaryConditionType')
        record.find('bcModel').text = 'Neumann'
        record.find('UseTemperature').text = 'false'
    else:
        raise ValueError('Unknown frozen rejection case: ' + case)
    for name, root in (('Model.xml', model), ('Materials.xml', material),
                       ('TransientTypeBC.xml', types)):
        members[name] = ET.tostring(root)
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, content in members.items():
            archive.writestr(name, content)
    return path


@pytest.mark.parametrize('entry', ('inspect', 'solve', 'run-new', 'run-existing'))
@pytest.mark.parametrize('case,code', REJECTIONS)
def test_declared_wet_rejection_never_publishes_a_partial_output(
        tmp_path, entry, case, code):
    """Each unsupported or inconsistent input fails at the public boundary."""
    source = _rejected_input(tmp_path / 'rejected.thmz', case)
    original = source.read_bytes()
    target = tmp_path / 'output.thmz'
    sentinel = b'Caller-owned archive must remain byte-identical on rejection.'
    if entry == 'run-existing':
        target.write_bytes(sentinel)
    if entry == 'inspect':
        report = inspect(source)
        assert not report.ok
        assert report.errors[0].code == code
    else:
        expected_type = UnsupportedError if code.startswith('THMZ-E5') \
            else ContractError
        with pytest.raises(expected_type) as caught:
            if entry == 'solve':
                solve(source)
            else:
                run_thmz(source, output=target)
        assert caught.value.code == code
    assert source.read_bytes() == original
    if entry == 'run-existing':
        assert target.read_bytes() == sentinel
    else:
        assert not target.exists()
    assert {path.name for path in tmp_path.iterdir()} <= \
        {'rejected.thmz', 'output.thmz'}
