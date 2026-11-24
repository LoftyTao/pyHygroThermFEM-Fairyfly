"""Public THMZ regressions for Ticket 07 material handling."""
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile

import pytest
from fairyfly_next_therm.result import THMZResult

from pyhygrothermfem_fairyfly import run_thmz


ASSETS = Path(__file__).parent / 'assets'
JUNCTION = ASSETS / 'ticket07_junction_80.thmz'
SERIES_WALL = ASSETS / 'ticket07_series_wall.thmz'
SLAB = ASSETS / 'slab.thmz'
ABS = 1e-6


def _members(path):
    with zipfile.ZipFile(path) as archive:
        return {name: archive.read(name) for name in archive.namelist()}


def _write(path, members):
    with zipfile.ZipFile(path, 'w') as archive:
        for name, content in members.items():
            archive.writestr(name, content)


def test_series_wall_public_run_matches_resistances_and_boundary_heat(tmp_path):
    output = tmp_path / 'series_wall_solved.thmz'
    result = THMZResult(run_thmz(SERIES_WALL, output=output))
    with zipfile.ZipFile(SERIES_WALL) as archive:
        mesh = ET.fromstring(archive.read('Mesh.xml'))
    nodes = mesh.findall('Case/Nodes/Node')

    heat_flux = 20 / 1.7
    x_min = min(float(node.findtext('x')) for node in nodes)
    assert len(result.temperatures) == len(nodes)
    assert result.heat_flux_magnitudes == pytest.approx(
        [heat_flux] * len(nodes), abs=ABS)
    for node, temperature in zip(nodes, result.temperatures):
        x = float(node.findtext('x')) - x_min
        resistance = 0.1 + (x if x <= 1 else 1 + (x - 1) / 2)
        assert temperature == pytest.approx(20 - heat_flux * resistance, abs=ABS)

    warm, cold = result.u_factors
    assert [warm.name, cold.name] == ['Warm Face', 'Cold Face']
    assert [warm.heat_flux, cold.heat_flux] == pytest.approx(
        [-2 * heat_flux, 2 * heat_flux], abs=ABS)
    for value in (warm, cold):
        assert value.delta_temperature == 20
        assert value.total_length == pytest.approx(2000, abs=ABS)
        assert value.total_u_factor == pytest.approx(heat_flux / 20, abs=ABS)


def test_wrong_material_region_preserves_existing_output(tmp_path):
    members = _members(JUNCTION)
    mesh = ET.fromstring(members['Mesh.xml'])
    element = next(item for item in mesh.findall('Case/Elements/Element')
                   if item.findtext('materialID') == '1')
    element.find('materialID').text = '2'
    members['Mesh.xml'] = ET.tostring(mesh, encoding='utf-8')
    source = tmp_path / 'wrong_region.thmz'
    _write(source, members)
    original = source.read_bytes()
    output = tmp_path / 'previous_result.thmz'
    previous = SLAB.read_bytes()
    output.write_bytes(previous)

    with pytest.raises(ValueError, match='Polygon|Element'):
        run_thmz(source, output=output)

    assert source.read_bytes() == original
    assert output.read_bytes() == previous


def test_single_material_legacy_id_42_keeps_public_results(tmp_path):
    members = _members(SLAB)
    mesh = ET.fromstring(members['Mesh.xml'])
    for element in mesh.findall('Case/Elements/Element'):
        element.find('materialID').text = '42'
    members['Mesh.xml'] = ET.tostring(mesh, encoding='utf-8')
    source = tmp_path / 'legacy_id_42.thmz'
    _write(source, members)

    result = THMZResult(run_thmz(source, output=tmp_path / 'legacy_solved.thmz'))
    assert result.temperatures == pytest.approx(
        [19.090909, 10, 0.909091] * 3, abs=ABS)
    assert result.heat_flux_magnitudes == pytest.approx([9.090909] * 9, abs=ABS)
    assert result.u_factors == ()
