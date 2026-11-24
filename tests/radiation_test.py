"""Scalar radiant environment through the public THMZ solve/read boundary."""
from pathlib import Path
from copy import deepcopy
import xml.etree.ElementTree as ET
import zipfile

import pytest
from fairyfly_next_therm.result import THMZResult
from pyhygrothermfem_fairyfly import run_thmz

from ufactor_test import tagged_slab


# Fixed before solving; six-decimal THMZ output, stricter than the V1 targets.
ABS = 1e-6  # C, W/m2, W/m, and W/(m2.K), as applicable.


def radiant_slab(path, flux=0, environment=0, surface=0.9,
                 radiant_temperature=20, film=10, view_factor=1):
    """Reuse the existing Q4 slab; change only boundary/material inputs."""
    tagged_slab(path)
    with zipfile.ZipFile(path) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    model = ET.fromstring(members['Model.xml'])
    conditions = ET.fromstring(members['SteadyStateBC.xml'])
    mesh = ET.fromstring(members['Mesh.xml'])
    hot_id = None
    for boundary in model.findall('Boundaries/Boundary'):
        boundary.find('ThermalEmissionProperties/Emissivity').text = str(surface)
        if boundary.findtext('Name') == 'Hot Air':
            hot_id = boundary.findtext('ID')
    for condition in conditions.findall('BoundaryCondition'):
        if condition.findtext('Name') == 'Hot Air':
            comp = condition.find('Comprehensive')
            comp.find('ConstantFlux/Flux').text = str(flux)
            comp.find('Convection/FilmCoefficient').text = str(film)
            radiation = comp.find('Radiation/BlackBodyRadiation')
            radiation.find('Temperature').text = str(radiant_temperature)
            radiation.find('Emissivity').text = str(environment)
            radiation.find('ViewFactor').text = str(view_factor)
    for bc in mesh.findall('Case/BCSegments/Convections/Convection'):
        if bc.findtext('label') == hot_id:
            bc.find('filmCoefficient').text = str(film)
    for name, root in (('Model.xml', model), ('SteadyStateBC.xml', conditions),
                       ('Mesh.xml', mesh)):
        members[name] = ET.tostring(root)
    with zipfile.ZipFile(path, 'w') as archive:
        for name, content in members.items():
            archive.writestr(name, content)
    return path


def test_added_flux_heats_slab_and_is_included_in_boundary_balance(tmp_path):
    source = radiant_slab(tmp_path / 'flux.thmz', flux=50)
    result = THMZResult(run_thmz(source))
    # Series resistance: q = (20 + 50/10) / (0.1 + 2 + 0.1).
    assert result.temperatures == pytest.approx(
        [23.86363636363636, 12.5, 1.136363636363636] * 3, abs=ABS)
    assert [q.x for q in result.heat_fluxes] == pytest.approx(
        [11.36363636363636] * 9, abs=ABS)
    hot, cold = result.u_factors
    assert [hot.heat_flux, cold.heat_flux] == pytest.approx(
        [-22.72727272727273, 22.72727272727273], abs=ABS)
    assert hot.heat_flux + cold.heat_flux == pytest.approx(0, abs=ABS)
    assert hot.total_u_factor == pytest.approx(0.568181818181818, abs=ABS)
    with zipfile.ZipFile(source) as before, zipfile.ZipFile(
            Path(source).with_name('flux_solved.thmz')) as after:
        for name in before.namelist():
            assert before.read(name) == after.read(name)


def slab_reference(environment, surface, radiant_temperature, flux=0,
                   film=10, view_factor=1):
    """Independent 1D resistance + gray-radiation energy balance, by bisection."""
    exchange = (1 / (1 / surface + 1 / environment + 1 / view_factor - 2)
                if surface and environment and view_factor else 0)
    low, high = -100.0, 100.0
    for _ in range(80):
        q = (low + high) / 2
        hot_surface = 2.1 * q
        incoming = (film * (20 - hot_surface) + flux + exchange * 5.6697e-8 *
                    ((radiant_temperature + 273.15) ** 4 -
                     (hot_surface + 273.15) ** 4))
        if incoming > q:
            low = q
        else:
            high = q
    return q, [2.1 * q, 1.1 * q, .1 * q] * 3


@pytest.mark.parametrize('environment,surface,rad_temp,flux,film,view', [
    (1, .9, 20, 0, 10, 1),
    (1, .9, 40, 0, 10, 1),
    (.5, .9, 40, 0, 10, 1),
    (1, .2, 40, 0, 10, 1),
    (1, .9, 40, 50, 10, 1),
    (1, .9, 40, 0, 0, 1),
    (0, .9, 40, 0, 10, 1),
    (1, 0, 40, 0, 10, 1),
    (1, .9, 40, 0, 10, 0),
    (1, .9, 40, 0, 10, .5),
])
def test_radiant_environment_matches_scalar_energy_balance(
        tmp_path, environment, surface, rad_temp, flux, film, view):
    source = radiant_slab(tmp_path / 'radiant.thmz', flux, environment,
                          surface, rad_temp, film, view)
    expected_flux, expected_temperatures = slab_reference(
        environment, surface, rad_temp, flux, film, view)
    result = THMZResult(run_thmz(source))
    assert result.temperatures == pytest.approx(expected_temperatures, abs=ABS)
    assert [q.x for q in result.heat_fluxes] == pytest.approx(
        [expected_flux] * 9, abs=ABS)
    hot, cold = result.u_factors
    assert [hot.heat_flux, cold.heat_flux] == pytest.approx(
        [-2 * expected_flux, 2 * expected_flux], abs=ABS)
    assert hot.heat_flux + cold.heat_flux == pytest.approx(0, abs=ABS)
    if film:
        assert hot.total_u_factor == pytest.approx(expected_flux / 20, abs=ABS)
    else:
        assert hot.total_u_factor is None  # Only one active air temperature.


def test_nonuniform_radiant_edge_balances_added_heat_and_cli_matches(tmp_path):
    from click.testing import CliRunner
    from pyhygrothermfem_fairyfly.cli import main

    source = radiant_slab(tmp_path / 'nonuniform.thmz', environment=1,
                          radiant_temperature=40)
    with zipfile.ZipFile(source) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    model = ET.fromstring(members['Model.xml'])
    conditions = ET.fromstring(members['SteadyStateBC.xml'])
    top_y = max(float(p.text) for p in model.findall(
        'Polygons/Polygon/Points/Point/y'))
    for boundary in model.findall('Boundaries/Boundary'):
        if all(float(boundary.findtext(p + '/y')) == top_y
               for p in ('StartPoint', 'EndPoint')):
            boundary.find('Name').text = 'Top Heat'
        boundary.find('FluxTag').text = boundary.findtext('Name')
    top = deepcopy(next(b for b in conditions.findall('BoundaryCondition')
                        if b.findtext('Name') == 'Hot Air'))
    top.find('Name').text = 'Top Heat'
    comp = top.find('Comprehensive')
    comp.find('Convection/Temperature').text = '0'
    comp.find('Convection/FilmCoefficient').text = '0'
    comp.find('ConstantFlux/Flux').text = '50'
    comp.find('Radiation/BlackBodyRadiation/Emissivity').text = '0'
    conditions.append(top)
    members['Model.xml'] = ET.tostring(model)
    members['SteadyStateBC.xml'] = ET.tostring(conditions)
    with zipfile.ZipFile(source, 'w') as archive:
        for name, content in members.items():
            archive.writestr(name, content)
    api = run_thmz(source)
    result = THMZResult(api)
    assert abs(result.temperatures[0] - result.temperatures[6]) > 1
    assert sum(u.heat_flux for u in result.u_factors) == pytest.approx(0, abs=ABS)
    assert next(u.heat_flux for u in result.u_factors
                if u.name == 'Top Heat') == pytest.approx(-100, abs=ABS)
    cli_path = tmp_path / 'cli.thmz'
    cli = CliRunner().invoke(main, ['run-thmz', str(source), '--output',
                                  str(cli_path)])
    assert cli.exit_code == 0, cli.output
    with zipfile.ZipFile(api) as first, zipfile.ZipFile(cli_path) as second:
        for name in ('SteadyStateMeshResults.xml', 'SteadyStateResults.xml'):
            assert first.read(name) == second.read(name)


@pytest.mark.parametrize('settings,diagnostic', [
    ({'environment': 1.1}, 'Emissivity'),
    ({'surface': -0.1}, 'Emissivity'),
    ({'view_factor': 1.1}, 'ViewFactor'),
    ({'radiant_temperature': -274}, 'absolute zero'),
    ({'flux': float('nan')}, 'finite'),
])
def test_invalid_radiant_inputs_preserve_existing_output(
        tmp_path, settings, diagnostic):
    source = radiant_slab(tmp_path / 'invalid.thmz', **settings)
    target = tmp_path / 'previous.thmz'
    target.write_bytes(b'previous result')
    original = source.read_bytes()
    with pytest.raises(ValueError, match=diagnostic):
        run_thmz(source, target)
    assert source.read_bytes() == original
    assert target.read_bytes() == b'previous result'


@pytest.mark.parametrize('field', ['Convection', 'ConstantFlux', 'Radiation'])
def test_conflicting_duplicate_boundary_terms_are_rejected(tmp_path, field):
    source = radiant_slab(tmp_path / 'duplicate.thmz', environment=1)
    with zipfile.ZipFile(source) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    conditions = ET.fromstring(members['SteadyStateBC.xml'])
    comp = next(c.find('Comprehensive') for c in conditions
                if c.findtext('Name') == 'Hot Air')
    duplicate = deepcopy(comp.find(field))
    next(node for node in duplicate.iter() if len(node) == 0).text = '25'
    comp.append(duplicate)
    members['SteadyStateBC.xml'] = ET.tostring(conditions)
    with zipfile.ZipFile(source, 'w') as archive:
        for name, content in members.items():
            archive.writestr(name, content)
    original = source.read_bytes()
    with pytest.raises(ValueError, match='duplicate'):
        run_thmz(source, source)
    assert source.read_bytes() == original
