"""Public E5017 diagnostics retain the active boundary's source values."""
import xml.etree.ElementTree as ET
import zipfile

import pytest

from pyhygrothermfem_fairyfly import UnsupportedError, inspect, run_thmz, solve
from ticket06_transient_dispatch_test import _transient_slab


@pytest.mark.parametrize('entry', ('inspect', 'solve', 'run_thmz'))
@pytest.mark.parametrize('kind', (
    'simple-film', 'comprehensive-film', 'flux', 'radiation', 'combined'))
def test_e5017_identifies_raw_active_fields(tmp_path, entry, kind):
    """All public rejection paths identify the source, not derived exchange."""
    source = _transient_slab(tmp_path, num_steps=2)
    with zipfile.ZipFile(source) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    model = ET.fromstring(members['Model.xml'])
    name = 'Cold Air' if kind == 'simple-film' else 'Adiabatic'
    boundaries = [b for b in model.findall('Boundaries/Boundary')
                  if b.findtext('Name') == name]
    for boundary in boundaries:
        data = boundary.find('TransientData')
        if data is not None:
            boundary.remove(data)
    bcs = ET.fromstring(members['SteadyStateBC.xml'])
    record = next(b for b in bcs.findall('BoundaryCondition')
                  if b.findtext('Name') == name)
    for child in list(record):
        if child.tag in ('Simplified', 'Comprehensive'):
            record.remove(child)
    expected = []
    if kind == 'simple-film':
        block = ET.SubElement(record, 'Simplified')
        ET.SubElement(block, 'Temperature').text = '20'
        ET.SubElement(block, 'FilmCoefficient').text = '1.250e-12'
        expected.append(('SteadyStateBC.xml/Simplified/FilmCoefficient',
                         '1.250e-12'))
    else:
        block = ET.SubElement(record, 'Comprehensive')
        convection = ET.SubElement(block, 'Convection')
        ET.SubElement(convection, 'Temperature').text = '20'
        film = '2.500' if kind in ('comprehensive-film', 'combined') else '0'
        ET.SubElement(convection, 'FilmCoefficient').text = film
        if film != '0':
            expected.append((
                'SteadyStateBC.xml/Comprehensive/Convection/FilmCoefficient',
                film))
        if kind in ('flux', 'combined'):
            ET.SubElement(ET.SubElement(block, 'ConstantFlux'), 'Flux').text = (
                '-2.500')
            expected.append(('SteadyStateBC.xml/Comprehensive/ConstantFlux/Flux',
                             '-2.500'))
        if kind in ('radiation', 'combined'):
            radiation = ET.SubElement(ET.SubElement(block, 'Radiation'),
                                      'BlackBodyRadiation')
            ET.SubElement(radiation, 'Temperature').text = '20'
            for tag, value in (('Emissivity', '0.800'), ('ViewFactor', '0.750')):
                ET.SubElement(radiation, tag).text = value
                expected.append((
                    'SteadyStateBC.xml/Comprehensive/Radiation/'
                    'BlackBodyRadiation/' + tag, value))
            for boundary in boundaries:
                boundary.find('ThermalEmissionProperties/Emissivity').text = (
                    '0.600')
            expected.append((
                'Model.xml/ThermalEmissionProperties/Emissivity', '0.600'))
    members['Model.xml'] = ET.tostring(model)
    members['SteadyStateBC.xml'] = ET.tostring(bcs)
    with zipfile.ZipFile(source, 'w') as archive:
        for member, content in members.items():
            archive.writestr(member, content)
    original = source.read_bytes()
    target = tmp_path / 'preserved.thmz'
    target.write_bytes(b'previous-output')
    if entry == 'inspect':
        report = inspect(source)
        assert not report.ok
        issue = report.errors[0]
        assert issue.code == 'THMZ-E5017'
        message = issue.message
    else:
        with pytest.raises(UnsupportedError) as caught:
            if entry == 'solve':
                solve(source)
            else:
                run_thmz(source, output=target)
        assert caught.value.code == 'THMZ-E5017'
        message = str(caught.value)
    assert 'Boundary ' + name in message
    assert any('ID ' + b.findtext('ID') in message for b in boundaries)
    for field, value in expected:
        assert "{}={!r}".format(field, value) in message
    assert source.read_bytes() == original
    assert target.read_bytes() == b'previous-output'
