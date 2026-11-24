"""Ticket 18 decision A: reject ambiguous transient boundary fallback."""
import xml.etree.ElementTree as ET
import zipfile

import pytest

from pyhygrothermfem_fairyfly import ContractError, UnsupportedError, inspect, run_thmz, solve
from ticket06_transient_dispatch_test import _transient_slab


def test_missing_transient_data_rejected_before_publication(tmp_path):
    """A cold convection record cannot silently drive a transient solve."""
    source = _transient_slab(tmp_path, num_steps=2)
    with zipfile.ZipFile(source) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    model = ET.fromstring(members['Model.xml'])
    cold = next(b for b in model.findall('Boundaries/Boundary')
                if b.findtext('Name') == 'Cold Air')
    data = cold.find('TransientData')
    if data is not None:
        cold.remove(data)
    members['Model.xml'] = ET.tostring(model)
    with zipfile.ZipFile(source, 'w') as archive:
        for name, content in members.items():
            archive.writestr(name, content)
    original = source.read_bytes()
    report = inspect(source)
    assert not report.ok
    assert report.errors[0].code == 'THMZ-E5017'
    for entry in (solve, run_thmz):
        with pytest.raises(ContractError) as caught:
            entry(source)
        assert caught.value.code == 'THMZ-E5017'
        assert 'Cold Air' in str(caught.value)
        assert 'TransientData' in str(caught.value)
    assert source.read_bytes() == original
    assert not source.with_name('transient_solved.thmz').exists()


@pytest.mark.parametrize('kind', ('tiny-film', 'flux', 'radiation'))
def test_zero_film_alone_does_not_exempt_active_boundary(tmp_path, kind):
    """Heat flux, radiation or humidity declarations are not sealed adiabatic."""
    source = _transient_slab(tmp_path, num_steps=2)
    with zipfile.ZipFile(source) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    bcs = ET.fromstring(members['SteadyStateBC.xml'])
    record = next(b for b in bcs.findall('BoundaryCondition')
                  if b.findtext('Name') == 'Adiabatic')
    record.remove(record.find('Simplified'))
    block = ET.SubElement(record, 'Comprehensive')
    convection = ET.SubElement(block, 'Convection')
    ET.SubElement(convection, 'Temperature').text = '20'
    ET.SubElement(convection, 'FilmCoefficient').text = (
        '1e-12' if kind == 'tiny-film' else '0')
    if kind == 'flux':
        ET.SubElement(ET.SubElement(block, 'ConstantFlux'), 'Flux').text = '1'
    elif kind == 'humidity':
        ET.SubElement(block, 'RelativeHumidity').text = '0'
    elif kind == 'radiation':
        radiation = ET.SubElement(ET.SubElement(block, 'Radiation'),
                                  'BlackBodyRadiation')
        for tag, value in (('Temperature', '20'), ('Emissivity', '0.9'),
                           ('ViewFactor', '1')):
            ET.SubElement(radiation, tag).text = value
    members['SteadyStateBC.xml'] = ET.tostring(bcs)
    with zipfile.ZipFile(source, 'w') as archive:
        for name, content in members.items():
            archive.writestr(name, content)
    target = tmp_path / 'preserved.thmz'
    target.write_bytes(b'previous-output')
    with pytest.raises(UnsupportedError) as caught:
        run_thmz(source, output=target)
    assert caught.value.code == 'THMZ-E5017'
    assert 'Adiabatic' in str(caught.value)
    assert target.read_bytes() == b'previous-output'


def test_explicit_constant_and_sealed_adiabatic_boundaries_are_accepted(tmp_path):
    """Explicit constants and zero-exchange exceptions remain supported."""
    source = _transient_slab(tmp_path, num_steps=2)
    assert inspect(source).ok
    result = solve(source)
    assert result.mode == 'dry-transient'
    assert result.steps_completed == 2


@pytest.mark.parametrize('wet', (False, True))
def test_default_adiabatic_humidity_is_inert_without_film(tmp_path, wet):
    """Legacy h=0, RH=0.5 adiabatic records do not exchange moisture."""
    if wet:
        from ticket09_hygrothermal_test import _wet_slab
        source = _wet_slab(tmp_path, num_steps=2)
    else:
        source = _transient_slab(tmp_path, num_steps=2)
    with zipfile.ZipFile(source) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    bcs = ET.fromstring(members['SteadyStateBC.xml'])
    record = next(b for b in bcs.findall('BoundaryCondition')
                  if b.findtext('Name') == 'Adiabatic')
    block = record.find('Simplified')
    if block is None:
        block = record.find('Comprehensive')
    humidity = block.find('RelativeHumidity')
    if humidity is None:
        humidity = ET.SubElement(block, 'RelativeHumidity')
    humidity.text = '0.5'
    members['SteadyStateBC.xml'] = ET.tostring(bcs)
    with zipfile.ZipFile(source, 'w') as archive:
        for name, content in members.items():
            archive.writestr(name, content)
    assert inspect(source).ok
    result = solve(source)
    assert result.steps_completed == 2
