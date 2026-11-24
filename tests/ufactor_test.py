"""Public U-factor contract: analytic film resistances and signed boundary heat."""
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile

import pytest
from fairyfly_next_therm.result import THMZResult
from pyhygrothermfem_fairyfly import run_thmz


SLAB = Path(__file__).parent / 'assets' / 'slab.thmz'
# Fixed before solving; the THMZ serializer writes six decimal places.
ABS = 1e-6  # degrees C, W/m, W/m2, W/(m2.K), or mm as applicable


def tagged_slab(path, same_tag=False, temperature=20, cold_temperature=0,
                rotated=False):
    """Transform the existing Q4 fixture; do not generate a new mesh."""
    with zipfile.ZipFile(SLAB) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    model = ET.fromstring(members['Model.xml'])
    mesh = ET.fromstring(members['Mesh.xml'])
    conditions = ET.fromstring(members['SteadyStateBC.xml'])
    origin = model.find('GlazingOrigin')
    ox, oy = float(origin.findtext('x')), float(origin.findtext('y'))

    def transform(x, y):
        y *= 2  # Two metre height makes W/m distinct from W/m2.
        return (0.8 * x - 0.6 * y, 0.6 * x + 0.8 * y) if rotated else (x, y)

    for point in (model.findall('Polygons/Polygon/Points/Point') +
                  model.findall('Boundaries/Boundary/StartPoint') +
                  model.findall('Boundaries/Boundary/EndPoint')):
        x, y = transform(float(point.findtext('x')) - ox,
                         float(point.findtext('y')) - oy)
        point.find('x').text, point.find('y').text = str(x + ox), str(y + oy)
    for node in mesh.findall('Case/Nodes/Node'):
        x, y = transform(float(node.findtext('x')), float(node.findtext('y')))
        node.find('x').text, node.find('y').text = str(x), str(y)
    air = {'Hot Air': temperature, 'Cold Air': cold_temperature}
    boundary_air = {}
    for boundary in model.findall('Boundaries/Boundary'):
        name = boundary.findtext('Name')
        if name in ('Hot Air', 'Cold Air'):
            boundary.find('FluxTag').text = 'Both' if same_tag else name
        if name in air:
            boundary_air[boundary.findtext('ID')] = air[name]
    for bc in mesh.findall('Case/BCSegments/Convections/Convection'):
        if bc.findtext('label') in boundary_air:
            value = str(boundary_air[bc.findtext('label')])
            bc.find('temperature1').text = bc.find('temperature2').text = value
    for bc in conditions.findall('BoundaryCondition'):
        if bc.findtext('Name') in air:
            bc.find('Comprehensive/Convection/Temperature').text = str(
                air[bc.findtext('Name')])
    for name, root in (('Model.xml', model), ('Mesh.xml', mesh),
                       ('SteadyStateBC.xml', conditions)):
        members[name] = ET.tostring(root)
    with zipfile.ZipFile(path, 'w') as archive:
        for name, content in members.items():
            archive.writestr(name, content)
    return path


def test_film_slab_u_factors_and_signed_integral(tmp_path):
    source = tagged_slab(tmp_path / 'tagged.thmz')
    result = THMZResult(run_thmz(source))
    assert result.u_factors is not None
    hot, cold = result.u_factors
    assert [u.name for u in result.u_factors] == ['Hot Air', 'Cold Air']
    assert [hot.heat_flux, cold.heat_flux] == pytest.approx(
        [-18.18181818181818, 18.18181818181818], abs=ABS)
    assert hot.heat_flux + cold.heat_flux == pytest.approx(0, abs=ABS)
    for value in (hot, cold):
        assert value.delta_temperature == 20
        assert value.total_length == pytest.approx(2000, abs=ABS)
        assert value.total_u_factor == pytest.approx(0.45454545454545, abs=ABS)
        assert value.projected_y_length == pytest.approx(2000, abs=ABS)
        assert value.projected_y_u_factor == value.total_u_factor
        assert value.projected_x_length == 0
        assert value.projected_x_u_factor is None
        assert value.projected_in_glass_plane_u_factor is None
        assert value.custom_rotation_u_factor is None
    assert result.temperatures == pytest.approx(
        [19.09090909090909, 10, 0.9090909090909091] * 3, abs=ABS)
    assert [q.x for q in result.heat_fluxes] == pytest.approx(
        [9.09090909090909] * 9, abs=ABS)


def test_same_tag_integrates_net_heat_not_absolute_flux(tmp_path):
    result = THMZResult(run_thmz(tagged_slab(
        tmp_path / 'both.thmz', same_tag=True)))
    assert len(result.u_factors) == 1
    value = result.u_factors[0]
    assert value.name == 'Both'
    assert value.total_length == 4000
    assert value.projected_y_length == 2000
    assert value.heat_flux == pytest.approx(0, abs=ABS)
    assert value.total_u_factor == pytest.approx(0, abs=ABS)


@pytest.mark.parametrize('temperature,cold_temperature', [(20, 20), (-20, 0)])
def test_zero_and_reversed_temperature_difference(
        tmp_path, temperature, cold_temperature):
    result = THMZResult(run_thmz(tagged_slab(
        tmp_path / 'temperature.thmz', temperature=temperature,
        cold_temperature=cold_temperature)))
    hot, cold = result.u_factors
    if temperature == cold_temperature:
        assert hot.delta_temperature == cold.delta_temperature == 0
        for value in (hot, cold):
            assert value.total_u_factor is None
            assert value.projected_y_u_factor is None
            assert value.total_length == 2000
            assert value.heat_flux == pytest.approx(0, abs=ABS)
    else:
        assert hot.heat_flux == pytest.approx(18.18181818181818, abs=ABS)
        assert cold.heat_flux == pytest.approx(-18.18181818181818, abs=ABS)
        assert hot.total_u_factor == pytest.approx(0.45454545454545, abs=ABS)


def test_rotated_projection_lengths_and_cli_match_api(tmp_path):
    from click.testing import CliRunner
    from pyhygrothermfem_fairyfly.cli import main

    source = tagged_slab(tmp_path / 'rotated.thmz', rotated=True)
    api = Path(run_thmz(source))
    target = tmp_path / 'cli.thmz'
    cli = CliRunner().invoke(main, ['run-thmz', str(source), '--output', str(target)])
    assert cli.exit_code == 0, cli.output
    with zipfile.ZipFile(api) as first, zipfile.ZipFile(target) as second:
        assert (first.read('SteadyStateResults.xml') ==
                second.read('SteadyStateResults.xml'))
    for value in THMZResult(str(target)).u_factors:
        assert value.total_length == pytest.approx(2000, abs=ABS)
        assert value.projected_x_length == pytest.approx(1200, abs=ABS)
        assert value.projected_y_length == pytest.approx(1600, abs=ABS)
        assert value.total_u_factor == pytest.approx(0.45454545454545, abs=ABS)
        assert value.projected_x_u_factor == pytest.approx(0.75757575757575, abs=ABS)
        assert value.projected_y_u_factor == pytest.approx(0.56818181818181, abs=ABS)


def test_no_tags_produces_empty_u_factor_case(tmp_path):
    result = THMZResult(run_thmz(SLAB, tmp_path / 'no_tags.thmz'))
    assert result.u_factors == ()


@pytest.mark.parametrize('record_tags', [
    [],
    ['Hot Air', 'Hot Air', 'Cold Air'],
    ['Hot Air', 'Cold Air', 'Unexpected'],
])
def test_tagged_input_rejects_incomplete_u_factor_rows(
        monkeypatch, tmp_path, record_tags):
    import pyhygrothermfem_fairyfly.run as runner
    import pylibraryfemtherm as fem

    source = tagged_slab(tmp_path / 'tagged.thmz')
    source_bytes = source.read_bytes()
    target = tmp_path / 'previous.thmz'
    target.write_bytes(SLAB.read_bytes())
    target_bytes = target.read_bytes()

    case = fem.SteadyStateResultsCase()
    case.model_type = fem.RunType.UFactor
    records = []
    for tag in record_tags:
        record = fem.SteadyStateUFactors()
        record.tag = tag
        records.append(record)
    case.steady_state_u_factors = records
    result = fem.SteadyStateResults()
    result.cases = [case]
    monkeypatch.setattr(
        runner, '_u_factor_results',
        lambda *args: fem.steady_state_results.to_string(result))

    with pytest.raises(RuntimeError, match='incomplete U-factor results'):
        run_thmz(source, output=target)

    assert source.read_bytes() == source_bytes
    assert target.read_bytes() == target_bytes
    assert not list(tmp_path.glob('.previous-*.thmz'))


def test_tagged_input_rejects_u_factor_xml_missing_heat_flux(
        monkeypatch, tmp_path):
    import pyhygrothermfem_fairyfly.run as runner

    source = tagged_slab(tmp_path / 'tagged.thmz')
    source_bytes = source.read_bytes()
    target = tmp_path / 'previous.thmz'
    target.write_bytes(SLAB.read_bytes())
    target_bytes = target.read_bytes()
    build_results = runner._u_factor_results

    def omit_heat_flux(*args):
        root = ET.fromstring(build_results(*args))
        record = root.find('Case/U-factors')
        record.remove(record.find('HeatFlux'))
        return ET.tostring(root, encoding='unicode')

    monkeypatch.setattr(runner, '_u_factor_results', omit_heat_flux)
    with pytest.raises(RuntimeError, match='incomplete U-factor results'):
        run_thmz(source, output=target)

    assert source.read_bytes() == source_bytes
    assert target.read_bytes() == target_bytes
    assert not list(tmp_path.glob('.previous-*.thmz'))


@pytest.mark.parametrize('fault', [
    'missing_total', 'unknown_projection', 'duplicate_projection',
    'missing_length', 'nan_length', 'negative_length', 'inf_u_factor',
])
def test_tagged_input_rejects_invalid_u_factor_projections(
        monkeypatch, tmp_path, fault):
    import copy
    import pyhygrothermfem_fairyfly.run as runner

    source = tagged_slab(tmp_path / 'tagged.thmz')
    source_bytes = source.read_bytes()
    target = tmp_path / 'previous.thmz'
    target.write_bytes(SLAB.read_bytes())
    target_bytes = target.read_bytes()
    build_results = runner._u_factor_results

    def malformed(*args):
        root = ET.fromstring(build_results(*args))
        record = root.find('Case/U-factors')
        total = next(p for p in record.findall('Projection')
                     if p.findtext('Length-type') == 'Total Length')
        if fault == 'missing_total':
            record.remove(total)
        elif fault == 'unknown_projection':
            unknown = copy.deepcopy(total)
            unknown.find('Length-type').text = 'bogus'
            record.append(unknown)
        elif fault == 'duplicate_projection':
            record.append(copy.deepcopy(total))
        elif fault == 'missing_length':
            total.remove(total.find('Length'))
        elif fault == 'nan_length':
            total.find('Length').text = 'NaN'
        elif fault == 'negative_length':
            total.find('Length').text = '-1'
        else:
            total.find('U-factor').text = 'Inf'
        return ET.tostring(root, encoding='unicode')

    monkeypatch.setattr(runner, '_u_factor_results', malformed)
    with pytest.raises(RuntimeError, match='incomplete U-factor results'):
        run_thmz(source, output=target)

    assert source.read_bytes() == source_bytes
    assert target.read_bytes() == target_bytes
    assert not list(tmp_path.glob('.previous-*.thmz'))
