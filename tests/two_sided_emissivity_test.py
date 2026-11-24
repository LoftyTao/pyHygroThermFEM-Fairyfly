"""Two-sided surface orientation through the public THMZ solve path."""
from pathlib import Path
import zipfile
import xml.etree.ElementTree as ET

import pytest
from fairyfly_next_therm.result import THMZResult
from pyhygrothermfem_fairyfly import run_thmz
from ufactor_test import tagged_slab


LOCAL_CAVITY = Path(__file__).parent / 'assets' / 'ticket06_local_cavity.thmz'
STEFAN_BOLTZMANN = 5.6697e-8


def cavity_emissivity_case(path, back, local):
    with zipfile.ZipFile(LOCAL_CAVITY) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    model = ET.fromstring(members['Model.xml'])
    materials = ET.fromstring(members['Materials.xml'])
    solid = next(item for item in materials.findall('Material')
                 if item.find('Solid') is not None)
    infrared = solid.find('Solid/Optical/Integrated/Infrared')
    infrared.find('Emissivity-Front').text = '0.2'
    infrared.find('Emissivity-Back').text = str(back)
    cavity = model.find('Cavities/Cavity')
    cavity.find('LocalEmissivities').text = 'true' if local else 'false'
    cavity.find('Emissivity1').text = '0.2'
    cavity.find('Emissivity2').text = str(back)
    for boundary in model.findall('Boundaries/Boundary'):
        if boundary.findtext('ID') not in ('13', '15'):
            continue
        side = 'Back' if boundary.findtext('ID') == '13' else 'Front'
        material_side = boundary.find('MaterialSide')
        if material_side is None:
            material_side = ET.SubElement(boundary, 'MaterialSide')
        material_side.text = side
    members['Model.xml'] = ET.tostring(model)
    members['Materials.xml'] = ET.tostring(materials)
    with zipfile.ZipFile(path, 'w') as archive:
        for name, content in members.items():
            archive.writestr(name, content)
    return path


def _results(path):
    result = THMZResult(run_thmz(path))
    return [*result.temperatures,
            *(value for flux in result.heat_fluxes for value in (flux.x, flux.y)),
            *(value for factor in result.u_factors
              for value in (factor.heat_flux, factor.total_u_factor))]


@pytest.mark.parametrize('back', (0.8, 0.4))
def test_frame_cavity_uses_each_neighbor_material_side(tmp_path, back):
    local = _results(cavity_emissivity_case(
        tmp_path / 'local.thmz', back, local=True))
    explicit = _results(cavity_emissivity_case(
        tmp_path / 'explicit.thmz', back, local=False))
    assert local == pytest.approx(explicit, abs=1e-6)


def test_changing_back_emissivity_changes_the_local_cavity_result(tmp_path):
    baseline = _results(cavity_emissivity_case(
        tmp_path / 'baseline.thmz', 0.8, local=True))
    changed = _results(cavity_emissivity_case(
        tmp_path / 'changed.thmz', 0.4, local=True))
    assert baseline != pytest.approx(changed, abs=1e-6)


def _two_sided_radiant_slab(path, front, back):
    tagged_slab(path)
    with zipfile.ZipFile(path) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    model = ET.fromstring(members['Model.xml'])
    conditions = ET.fromstring(members['SteadyStateBC.xml'])
    materials = ET.fromstring(members['Materials.xml'])
    material = next(item for item in materials.findall('Material')
                    if item.find('Solid') is not None)
    infrared = material.find('Solid/Optical/Integrated/Infrared')
    infrared.find('Emissivity-Front').text = str(front)
    infrared.find('Emissivity-Back').text = str(back)
    surfaces = {'Hot Air': ('Front', front, 35),
                'Cold Air': ('Back', back, -15)}
    for boundary in model.findall('Boundaries/Boundary'):
        name = boundary.findtext('Name')
        if name not in surfaces:
            continue
        side, emissivity, _ = surfaces[name]
        material_side = boundary.find('MaterialSide')
        if material_side is None:
            material_side = ET.SubElement(boundary, 'MaterialSide')
        material_side.text = side
        boundary.find('ThermalEmissionProperties/Emissivity').text = str(emissivity)
    for condition in conditions.findall('BoundaryCondition'):
        name = condition.findtext('Name')
        if name not in surfaces:
            continue
        _, _, radiant_temperature = surfaces[name]
        radiation = condition.find('Comprehensive/Radiation/BlackBodyRadiation')
        radiation.find('Temperature').text = str(radiant_temperature)
        radiation.find('Emissivity').text = '1'
        radiation.find('ViewFactor').text = '1'
    members['Model.xml'] = ET.tostring(model)
    members['Materials.xml'] = ET.tostring(materials)
    members['SteadyStateBC.xml'] = ET.tostring(conditions)
    with zipfile.ZipFile(path, 'w') as archive:
        for name, content in members.items():
            archive.writestr(name, content)
    return path


def _bisect(function, low, high):
    low_value = function(low)
    assert low_value * function(high) < 0
    for _ in range(80):
        middle = (low + high) / 2
        middle_value = function(middle)
        if low_value * middle_value <= 0:
            high = middle
        else:
            low, low_value = middle, middle_value
    return (low + high) / 2


def _radiant_slab_reference(front, back):
    def hot_surface(heat):
        def balance(temperature):
            return (10 * (20 - temperature) + front * STEFAN_BOLTZMANN *
                    ((35 + 273.15) ** 4 - (temperature + 273.15) ** 4) - heat)
        return _bisect(balance, -100, 100)

    def cold_surface(heat):
        def balance(temperature):
            return (10 * temperature + back * STEFAN_BOLTZMANN *
                    ((temperature + 273.15) ** 4 - (-15 + 273.15) ** 4) - heat)
        return _bisect(balance, -100, 100)

    heat = _bisect(lambda value: hot_surface(value) - cold_surface(value) -
                   2 * value, -100, 100)
    hot, cold = hot_surface(heat), cold_surface(heat)
    return heat, [hot, hot - heat, cold] * 3


@pytest.mark.parametrize('front,back', ((0.2, 0.8), (0.4, 0.8), (0.2, 0.4)))
def test_front_and_back_external_radiation_match_independent_balance(
        tmp_path, front, back):
    result = THMZResult(run_thmz(_two_sided_radiant_slab(
        tmp_path / 'radiant.thmz', front, back)))
    heat, temperatures = _radiant_slab_reference(front, back)
    assert result.temperatures == pytest.approx(temperatures, abs=1e-6)
    assert [flux.x for flux in result.heat_fluxes] == pytest.approx(
        [heat] * 9, abs=1e-6)
    hot, cold = result.u_factors
    assert [hot.heat_flux, cold.heat_flux] == pytest.approx(
        [-2 * heat, 2 * heat], abs=1e-6)
    assert [hot.total_u_factor, cold.total_u_factor] == pytest.approx(
        [heat / 20, heat / 20], abs=1e-6)


@pytest.mark.parametrize('wrong_neighbor', ('missing', 'other_solid'))
def test_frame_cavity_neighbor_must_match_mesh_topology_and_not_publish(
        tmp_path, wrong_neighbor):
    source = cavity_emissivity_case(tmp_path / 'wrong neighbor.thmz',
                                    0.8, local=True)
    with zipfile.ZipFile(source) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    model = ET.fromstring(members['Model.xml'])
    boundary = next(item for item in model.findall('Boundaries/Boundary')
                    if item.findtext('ID') == '13')
    neighbor2 = boundary.find('NeighborPolygonUUID2')
    if wrong_neighbor == 'missing':
        boundary.remove(neighbor2)
    else:
        gas_uuid = model.find('Polygons/Polygon[CavityUUID]').findtext('UUID')
        other_solid = next(item.findtext('UUID')
                           for item in model.findall('Polygons/Polygon')
                           if item.findtext('UUID') != gas_uuid
                           and item.findtext('UUID') != neighbor2.text)
        neighbor2.text = other_solid
    members['Model.xml'] = ET.tostring(model)
    with zipfile.ZipFile(source, 'w') as archive:
        for name, content in members.items():
            archive.writestr(name, content)
    target = tmp_path / 'not-published.thmz'
    with pytest.raises(ValueError, match=r'Frame Cavity boundary 13.*neighbor'):
        run_thmz(source, target)
    assert not target.exists()


def test_local_cavity_mesh_edges_need_complete_boundary_coverage(tmp_path):
    source = cavity_emissivity_case(tmp_path / 'partial cavity boundary.thmz',
                                    0.8, local=True)
    with zipfile.ZipFile(source) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    model = ET.fromstring(members['Model.xml'])
    boundary = next(item for item in model.findall('Boundaries/Boundary')
                    if item.findtext('ID') == '13')
    boundary.find('EndPoint/y').text = '-400'
    members['Model.xml'] = ET.tostring(model)
    with zipfile.ZipFile(source, 'w') as archive:
        for name, content in members.items():
            archive.writestr(name, content)
    target = tmp_path / 'not-published.thmz'
    with pytest.raises(ValueError, match=r'Local cavity .*mesh edge .*Frame Cavity coverage'):
        run_thmz(source, target)
    assert not target.exists()


@pytest.mark.parametrize('side', ('Inside', ''))
def test_invalid_boundary_material_side_has_boundary_diagnostic(tmp_path, side):
    source = cavity_emissivity_case(tmp_path / 'invalid.thmz', 0.8, local=True)
    with zipfile.ZipFile(source) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    model = ET.fromstring(members['Model.xml'])
    model.find('Boundaries/Boundary/MaterialSide').text = side
    members['Model.xml'] = ET.tostring(model)
    with zipfile.ZipFile(source, 'w') as archive:
        for name, content in members.items():
            archive.writestr(name, content)
    with pytest.raises(ValueError, match='MaterialSide must be Front or Back'):
        run_thmz(source)
