"""Public THMZ contract and the single-material analytic slab benchmark."""
from pathlib import Path
import json
import zipfile
import xml.etree.ElementTree as ET

import pytest
from fairyfly_next_therm.result import THMZResult

from pyhygrothermfem_fairyfly import run_thmz


SLAB = Path(__file__).parent / 'assets' / 'slab.thmz'
# Fixed before the first solve: LibraryFEMTherm writes six decimal places.
TEMPERATURE_ABS = 1e-6  # degrees Celsius
FLUX_ABS = 1e-6  # W/m2
RELATIVE = 1e-8


def test_slab_api_returns_readable_analytic_results(tmp_path):
    """The actual kernel solves a slab exported by the baseline Fairyfly."""
    original = SLAB.read_bytes()
    target = tmp_path / '单材料 slab.thmz'
    result = Path(run_thmz(SLAB, output=target))
    assert result == target
    assert SLAB.read_bytes() == original
    actual = THMZResult(str(result))
    expected = [19.09090909090909, 10.0, 0.9090909090909091] * 3
    assert actual.temperatures == pytest.approx(
        expected, abs=TEMPERATURE_ABS, rel=RELATIVE)
    assert [v.x for v in actual.heat_fluxes] == pytest.approx(
        [9.09090909090909] * 9, abs=FLUX_ABS, rel=RELATIVE)
    assert [v.y for v in actual.heat_fluxes] == pytest.approx(
        [0.0] * 9, abs=FLUX_ABS, rel=RELATIVE)
    assert len(actual.mesh.faces) == 4
    assert [(v.x, v.y) for v in actual.mesh.vertices] == [
        (0, 0), (1000, 0), (2000, 0),
        (0, 500), (1000, 500), (2000, 500),
        (0, 1000), (1000, 1000), (2000, 1000)]
    with zipfile.ZipFile(SLAB) as source, zipfile.ZipFile(result) as solved:
        for name in source.namelist():
            assert solved.read(name) == source.read(name)


def test_cli_matches_api_and_native_reference(tmp_path):
    """Both public entry points match the separately built native executable."""
    from click.testing import CliRunner
    from pyhygrothermfem_fairyfly.cli import main

    reference = json.loads(SLAB.with_name('native_reference.json').read_text())
    folder = tmp_path / '中文 folder'
    folder.mkdir()
    source = folder / 'model input.thmz'
    source.write_bytes(SLAB.read_bytes())
    target = folder / 'cli result.thmz'
    cli = CliRunner().invoke(main, [
        'run-thmz', str(source), '--output', str(target)])
    assert cli.exit_code == 0, cli.output
    assert Path(cli.stdout.strip()) == target
    api_result = THMZResult(run_thmz(source, folder / 'api result.thmz'))
    cli_result = THMZResult(str(target))
    assert cli_result.temperatures == api_result.temperatures
    assert cli_result.heat_fluxes == api_result.heat_fluxes
    assert cli_result.temperatures == pytest.approx(
        reference['temperature'], abs=TEMPERATURE_ABS, rel=RELATIVE)
    for actual, expected in zip(cli_result.heat_fluxes, reference['heat_flux']):
        assert (actual.x, actual.y) == pytest.approx(
            expected, abs=FLUX_ABS, rel=RELATIVE)


def test_atomic_rerun_replaces_stale_results(tmp_path):
    """In-place reruns use new material values and discard old computed data."""
    target = tmp_path / 'in place.thmz'
    run_thmz(SLAB, output=target)
    with zipfile.ZipFile(target) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    root = ET.fromstring(members['Materials.xml'])
    root.find('Material/Solid/HygroThermal/ThermalConductivityDry').text = '2'
    members['Materials.xml'] = ET.tostring(root)
    members['SteadyStateResults.xml'] = b'<stale U-factor/>'
    members['notes-about-results.txt'] = b'User notes must survive.'
    with zipfile.ZipFile(target, 'w') as archive:
        for name, content in members.items():
            archive.writestr(name, content)
    assert Path(run_thmz(target, output=target)) == target
    result = THMZResult(str(target))
    assert result.heat_flux_magnitudes == pytest.approx(
        [16.66666666666667] * 9, abs=FLUX_ABS, rel=RELATIVE)
    assert result.u_factors == ()
    with zipfile.ZipFile(target) as archive:
        assert archive.read('notes-about-results.txt') == members[
            'notes-about-results.txt']


def test_failed_result_publication_preserves_source_and_target(
        monkeypatch, tmp_path):
    """An output replacement failure keeps both existing archives intact."""
    import pyhygrothermfem_fairyfly.run as runner

    source = tmp_path / '输入 with spaces.thmz'
    source.write_bytes(SLAB.read_bytes())
    target = tmp_path / 'existing.thmz'
    target.write_bytes(SLAB.read_bytes())
    before_source, before_target = source.read_bytes(), target.read_bytes()

    def fail_replace(*args):
        raise OSError('disk full at publication')

    with monkeypatch.context() as patch:
        patch.setattr(runner.os, 'replace', fail_replace)
        with pytest.raises(OSError, match='disk full'):
            run_thmz(source, target)
    assert source.read_bytes() == before_source
    assert target.read_bytes() == before_target
    assert not list(tmp_path.glob('.existing-*.thmz'))


def test_nonconverged_kernel_preserves_existing_result(monkeypatch, tmp_path):
    """The public solve never labels a kernel failure as a new result."""
    from pyhygrothermfem_fairyfly import _native

    source = tmp_path / 'model.thmz'
    source.write_bytes(SLAB.read_bytes())
    target = tmp_path / 'previous.thmz'
    target.write_bytes(SLAB.read_bytes())
    original = source.read_bytes()

    def fail_solve(*args, **kwargs):
        raise RuntimeError('kernel did not converge at assembly')

    monkeypatch.setattr(_native, 'solve_steady', fail_solve)
    with pytest.raises(RuntimeError, match='did not converge'):
        run_thmz(source, target)
    assert source.read_bytes() == original
    assert target.read_bytes() == original


def test_corrupt_thmz_is_diagnostic_and_preserves_previous_result(tmp_path):
    """Corrupt ZIP data reports the archive stage and keeps old output."""
    source = tmp_path / 'broken.thmz'
    source.write_bytes(b'not a THMZ ZIP')
    target = tmp_path / 'previous.thmz'
    target.write_bytes(SLAB.read_bytes())
    previous = target.read_bytes()
    with pytest.raises(ValueError, match='Invalid THMZ ZIP archive'):
        run_thmz(source, target)
    assert source.read_bytes() == b'not a THMZ ZIP'
    assert target.read_bytes() == previous


@pytest.mark.parametrize('problem,diagnostic', [
    ('missing_mesh', 'Mesh.xml'),
    ('missing_conductivity', 'ThermalConductivityDry'),
    ('bad_node', 'node indices'),
    ('nan_conductivity', 'finite'),
    ('unsupported_moisture', 'hygrothermal solves require Transient'),
    ('wrong_boundary', '[Bb]oundary'),
    ('conflicting_film', '[Cc]onvection'),
    ('invalid_flux', '[Ff]lux'),
    ('invalid_radiation', '[Rr]adiation'),
    ('wrong_polygon', '[Pp]olygon'),
    ('boundary_gap', '[Bb]oundary'),
])
def test_invalid_input_preserves_input_and_existing_result(
        tmp_path, problem, diagnostic):
    """Malformed/unsupported THMZ inputs never publish or return stale results."""
    with zipfile.ZipFile(SLAB) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    if problem == 'missing_mesh':
        members.pop('Mesh.xml')
    elif problem in ('missing_conductivity', 'nan_conductivity'):
        root = ET.fromstring(members['Materials.xml'])
        hygro = root.find('Material/Solid/HygroThermal')
        if problem == 'missing_conductivity':
            hygro.remove(hygro.find('ThermalConductivityDry'))
        else:
            hygro.find('ThermalConductivityDry').text = 'NaN'
        members['Materials.xml'] = ET.tostring(root)
    elif problem == 'bad_node':
        root = ET.fromstring(members['Mesh.xml'])
        root.find('Case/Elements/Element/node1').text = '9999'
        members['Mesh.xml'] = ET.tostring(root)
    elif problem == 'unsupported_moisture':
        root = ET.fromstring(members['Model.xml'])
        root.find('Properties/CalculationOptions/SimulateMoisture').text = 'true'
        members['Model.xml'] = ET.tostring(root)
    elif problem in ('wrong_polygon', 'boundary_gap'):
        root = ET.fromstring(members['Model.xml'])
        if problem == 'wrong_polygon':
            points = root.findall('Polygons/Polygon/Points/Point/x')
            right = max(float(p.text) for p in points)
            for point in points:
                if float(point.text) == right:
                    point.text = str(right + 1000)
        else:
            point = root.find('Boundaries/Boundary/StartPoint/y')
            point.text = str(float(point.text) + 500)
        members['Model.xml'] = ET.tostring(root)
    elif problem in ('wrong_boundary', 'conflicting_film'):
        root = ET.fromstring(members['Mesh.xml'])
        boundary = root.find('Case/BCSegments/Convections/Convection')
        if problem == 'wrong_boundary':
            boundary.find('label').text = '3'  # Cold boundary on the hot edge.
        else:
            boundary.find('filmCoefficient').text = '12'
        members['Mesh.xml'] = ET.tostring(root)
    else:
        root = ET.fromstring(members['SteadyStateBC.xml'])
        comprehensive = root.find('BoundaryCondition/Comprehensive')
        if problem == 'invalid_flux':
            comprehensive.find('ConstantFlux/Flux').text = 'NaN'
        else:
            comprehensive.find('Radiation/BlackBodyRadiation/Emissivity').text = '1.1'
            model = ET.fromstring(members['Model.xml'])
            model.find('Boundaries/Boundary/ThermalEmissionProperties/Emissivity') \
                .text = '.9'
            members['Model.xml'] = ET.tostring(model)
        members['SteadyStateBC.xml'] = ET.tostring(root)
    source = tmp_path / 'invalid.thmz'
    with zipfile.ZipFile(source, 'w') as archive:
        for name, content in members.items():
            archive.writestr(name, content)
    original = source.read_bytes()
    target = tmp_path / 'previous.thmz'
    previous = SLAB.read_bytes()
    target.write_bytes(previous)
    with pytest.raises(ValueError, match=diagnostic):
        run_thmz(source, output=target)
    assert source.read_bytes() == original
    assert target.read_bytes() == previous
    assert not list(tmp_path.glob('.*.thmz'))


def test_cli_failure_is_nonzero_and_diagnostic(tmp_path):
    """The command reports invalid input without a success path on stdout."""
    from click.testing import CliRunner
    from pyhygrothermfem_fairyfly.cli import main

    source = tmp_path / 'missing_mesh.thmz'
    with zipfile.ZipFile(SLAB) as archive, zipfile.ZipFile(source, 'w') as target:
        for name in archive.namelist():
            if name != 'Mesh.xml':
                target.writestr(name, archive.read(name))
    result = CliRunner().invoke(main, ['run-thmz', str(source)])
    assert result.exit_code != 0
    assert 'Mesh.xml' in result.stderr
    assert result.stdout == ''
    assert not source.with_name('missing_mesh_solved.thmz').exists()


def test_material_key_is_independent_of_polygon_id(tmp_path):
    """Changing a polygon label must not change single-material physics."""
    source = tmp_path / 'polygon42.thmz'
    with zipfile.ZipFile(SLAB) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    model = ET.fromstring(members['Model.xml'])
    model.find('Polygons/Polygon/ID').text = '42'
    members['Model.xml'] = ET.tostring(model)
    with zipfile.ZipFile(source, 'w') as archive:
        for name, content in members.items():
            archive.writestr(name, content)
    result = THMZResult(run_thmz(source))
    assert result.heat_flux_magnitudes == pytest.approx(
        [9.09090909090909] * 9, abs=FLUX_ABS, rel=RELATIVE)
