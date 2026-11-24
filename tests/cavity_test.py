"""Public THMZ cavity behavior on a premeshed two-region wall."""
import json
from pathlib import Path
import zipfile
import xml.etree.ElementTree as ET

import pytest

from fairyfly_next_therm.result import THMZResult
from pyhygrothermfem_fairyfly import run_thmz


WALL = Path(__file__).parent / 'assets' / 'ticket07_series_wall.thmz'
LOCAL_CAVITY = Path(__file__).parent / 'assets' / 'ticket06_local_cavity.thmz'


def cavity_wall(tmp_path, standard='CEN', gas='Air', gravity='Down',
                emissivities=('0.9', '0.9')):
    with zipfile.ZipFile(WALL) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    model = ET.fromstring(members['Model.xml'])
    material_uuid = model.find('Polygons/Polygon/MaterialUUID').text
    model.find('Properties/ModelExposure/GravityOrientation').text = gravity
    members['Model.xml'] = ET.tostring(model)
    materials = ET.fromstring(members['Materials.xml'])
    material = next(item for item in materials.findall('Material')
                    if item.findtext('UUID') == material_uuid)
    material.remove(material.find('Solid'))
    cavity = ET.SubElement(material, 'Cavity')
    for key, value in (('CavityStandard', standard), ('Gas', gas),
                       ('EmissivitySide1', emissivities[0]),
                       ('EmissivitySide2', emissivities[1])):
        ET.SubElement(cavity, key).text = value
    members['Materials.xml'] = ET.tostring(materials)
    path = tmp_path / '{}-{}-{}.thmz'.format(standard, gas, gravity)
    with zipfile.ZipFile(path, 'w') as archive:
        for name, content in members.items():
            archive.writestr(name, content)
    return path


@pytest.mark.parametrize('standard', ('CEN', 'ISO15099',
                                      'ISO15099Ventilated'))
@pytest.mark.parametrize('gas', ('Air', 'Argon', 'Krypton', 'Xenon'))
def test_supported_cavity_matrix_solves_and_balances(tmp_path, standard, gas):
    result = THMZResult(run_thmz(cavity_wall(tmp_path, standard, gas)))
    warm, cold = result.u_factors
    assert len(result.temperatures) > 0
    assert warm.total_u_factor > 0
    assert warm.heat_flux + cold.heat_flux == pytest.approx(0, abs=1e-6)
    assert warm.total_u_factor == pytest.approx(cold.total_u_factor, abs=1e-6)


def test_iso_gas_gravity_and_emissivity_affect_heat_flow(tmp_path):
    def u(gas='Air', gravity='Down', emissivities=('0.9', '0.9')):
        return THMZResult(run_thmz(cavity_wall(
            tmp_path, 'ISO15099', gas, gravity, emissivities))).u_factors[
                0].total_u_factor

    air = u()
    assert air > u(gas='Argon') > u(gas='Krypton') > u(gas='Xenon')
    assert u(gravity='Left') != pytest.approx(air, abs=1e-3)
    assert u(gravity='Right') != pytest.approx(air, abs=1e-3)
    assert u(emissivities=('0.2', '0.2')) < air


@pytest.mark.parametrize('gas', ('Air', 'Argon', 'Helium'))
def test_unmapped_nfrc_xml_fails_without_publishing(tmp_path, gas):
    source = cavity_wall(tmp_path, standard='NFRC', gas=gas)
    with pytest.raises(ValueError, match='NFRC.*Fairyfly-Next'):
        run_thmz(source)
    assert not source.with_name(source.stem + '_solved.thmz').exists()


def test_next_exported_nfrc_metadata_reaches_result(tmp_path):
    source = cavity_wall(tmp_path, standard='ISO15099')
    with zipfile.ZipFile(source) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    material_uuid = ET.fromstring(members['Model.xml']).findtext(
        'Polygons/Polygon/MaterialUUID')
    mapping = {
        'material_uuid': material_uuid,
        'requested': 'NFRC', 'effective': 'ISO15099',
        'reference': 'LBNL NFRC Simulation Manual 6.3.6',
    }
    members['FairyflyNext.json'] = json.dumps({
        'cavity_model_mappings': [mapping]}).encode('utf-8')
    with zipfile.ZipFile(source, 'w') as archive:
        for name, content in members.items():
            archive.writestr(name, content)

    result = run_thmz(source)
    with zipfile.ZipFile(result) as archive:
        assert json.loads(archive.read('pyHygroTherm.json'))[
            'cavity_model_mappings'] == [mapping]
        assert archive.read('FairyflyNext.json') == members['FairyflyNext.json']

    mapping['material_uuid'] = 'unknown-material'
    members['FairyflyNext.json'] = json.dumps({
        'cavity_model_mappings': [mapping]}).encode('utf-8')
    with zipfile.ZipFile(source, 'w') as archive:
        for name, content in members.items():
            archive.writestr(name, content)
    with pytest.raises(ValueError, match='FairyflyNext.json'):
        run_thmz(source, tmp_path / 'invalid-result.thmz')
    assert not (tmp_path / 'invalid-result.thmz').exists()

    mapping['material_uuid'] = material_uuid
    materials = ET.fromstring(members['Materials.xml'])
    material = next(item for item in materials.findall('Material')
                    if item.findtext('UUID') == material_uuid)
    material.find('Cavity/Gas').text = 'Argon'
    members['Materials.xml'] = ET.tostring(materials)
    members['FairyflyNext.json'] = json.dumps({
        'cavity_model_mappings': [mapping]}).encode('utf-8')
    with zipfile.ZipFile(source, 'w') as archive:
        for name, content in members.items():
            archive.writestr(name, content)
    with pytest.raises(ValueError, match='FairyflyNext.json'):
        run_thmz(source, tmp_path / 'invalid-gas-result.thmz')
    assert not (tmp_path / 'invalid-gas-result.thmz').exists()


def test_local_cavity_radiation_uses_neighbor_solid_emissivity(tmp_path):
    baseline = THMZResult(run_thmz(LOCAL_CAVITY, tmp_path / 'baseline.thmz'))
    with zipfile.ZipFile(LOCAL_CAVITY) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    original_materials = members['Materials.xml']
    materials = ET.fromstring(members['Materials.xml'])
    for solid in materials.findall('Material/Solid'):
        infrared = solid.find('Optical/Integrated/Infrared')
        infrared.find('Emissivity-Front').text = '0.9'
        infrared.find('Emissivity-Back').text = '0.9'
    members['Materials.xml'] = ET.tostring(materials)
    source = tmp_path / 'high emissivity.thmz'
    with zipfile.ZipFile(source, 'w') as archive:
        for name, content in members.items():
            archive.writestr(name, content)
    radiating = THMZResult(run_thmz(source))
    assert baseline.u_factors[0].total_u_factor > 0
    assert radiating.u_factors[0].total_u_factor > \
        baseline.u_factors[0].total_u_factor + 0.1

    model = ET.fromstring(members['Model.xml'])
    cavity = model.find('Cavities/Cavity')
    cavity.find('LocalEmissivities').text = 'false'
    cavity.find('Emissivity1').text = '0.2'
    cavity.find('Emissivity2').text = '0.2'
    members['Materials.xml'] = original_materials
    members['Model.xml'] = ET.tostring(model)
    override = tmp_path / 'explicit cavity emissivity.thmz'
    with zipfile.ZipFile(override, 'w') as archive:
        for name, content in members.items():
            archive.writestr(name, content)
    explicit = THMZResult(run_thmz(override))
    assert explicit.u_factors[0].total_u_factor > \
        baseline.u_factors[0].total_u_factor + 0.01


def test_iso_uses_cavity_pressure_from_thmz(tmp_path):
    baseline = THMZResult(run_thmz(LOCAL_CAVITY, tmp_path / 'normal.thmz'))
    with zipfile.ZipFile(LOCAL_CAVITY) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    model = ET.fromstring(members['Model.xml'])
    model.find('Cavities/Cavity/Pressure').text = '50000'
    members['Model.xml'] = ET.tostring(model)
    source = tmp_path / 'low pressure.thmz'
    with zipfile.ZipFile(source, 'w') as archive:
        for name, content in members.items():
            archive.writestr(name, content)
    reduced = THMZResult(run_thmz(source))
    assert abs(baseline.u_factors[0].total_u_factor -
               reduced.u_factors[0].total_u_factor) > 0.001
