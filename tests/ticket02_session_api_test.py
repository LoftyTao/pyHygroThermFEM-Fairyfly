"""Ticket 02 session API: inspect, solve, coded exceptions, CLI plumbing."""
from hashlib import sha256
from pathlib import Path
import json
import zipfile
import xml.etree.ElementTree as ET

import pytest

import pyhygrothermfem_fairyfly
from pyhygrothermfem_fairyfly import (ContractError, MeshError, SolveError, THMZError,
                          UnsupportedError, inspect as inspect_thmz, run_thmz,
                          solve)

SLAB = Path(__file__).parent / 'assets' / 'slab.thmz'
ABS = 1e-6  # Serializer writes six decimal places.


def test_inspect_valid_archive_reports_members_and_mode():
    report = inspect_thmz(SLAB)
    assert report.ok
    assert report.errors == ()
    assert set(report.members) == {
        'Materials.xml', 'Gases.xml', 'Model.xml', 'SteadyStateBC.xml',
        'Mesh.xml'}
    assert report.mode == 'Steady-State'
    assert report.contract_version is None
    assert report.solver_version is None
    assert report.input_sha256 == sha256(SLAB.read_bytes()).hexdigest()


def test_inspect_returns_report_for_corrupt_and_contract_failures(tmp_path):
    corrupt = tmp_path / 'corrupt.thmz'
    corrupt.write_bytes(b'not a THMZ ZIP')
    report = inspect_thmz(corrupt)
    assert not report.ok
    assert [issue.code for issue in report.errors] == ['THMZ-E1006']

    with zipfile.ZipFile(SLAB) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    members.pop('Mesh.xml')
    missing = tmp_path / 'missing.thmz'
    with zipfile.ZipFile(missing, 'w') as archive:
        for name, content in members.items():
            archive.writestr(name, content)
    report = inspect_thmz(missing)
    assert not report.ok
    assert [issue.code for issue in report.errors] == ['THMZ-E1004']
    assert 'Mesh.xml' in report.errors[0].message
    assert report.members  # Member listing still available.


def test_inspect_reports_newer_contract_major(tmp_path):
    with zipfile.ZipFile(SLAB) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    members['pyHygroTherm.json'] = json.dumps({
        'contract_version': 3000}).encode('utf-8')
    source = tmp_path / 'newer.thmz'
    with zipfile.ZipFile(source, 'w') as archive:
        for name, content in members.items():
            archive.writestr(name, content)
    report = inspect_thmz(source)
    assert not report.ok
    assert [issue.code for issue in report.errors] == ['THMZ-E1102']


def test_inspect_raises_oserror_for_unreadable_input(tmp_path):
    with pytest.raises(OSError):
        inspect_thmz(tmp_path / 'does not exist.thmz')


def test_solve_returns_lazy_result_handle():
    result = solve(SLAB)
    assert isinstance(result, pyhygrothermfem_fairyfly.SolvedResult)
    assert result.mode == 'dry-steady'
    assert list(result.temperatures) == pytest.approx(
        [19.09090909090909, 10.0, 0.9090909090909091] * 3, abs=ABS)
    assert [flux[0] for flux in result.heat_fluxes] == pytest.approx(
        [9.09090909090909] * 9, abs=ABS)
    assert [flux[1] for flux in result.heat_fluxes] == pytest.approx(
        [0.0] * 9, abs=ABS)
    assert result.energy_error_percent is not None
    assert result.provenance['contract_version'] == 2001
    assert result.provenance['solver_version'] == 1
    assert result.provenance['input_sha256'] == sha256(
        SLAB.read_bytes()).hexdigest()
    assert result.provenance['mode'] == 'dry-steady'
    members = result.members
    with zipfile.ZipFile(SLAB) as archive:
        for name in archive.namelist():
            assert members[name] == archive.read(name)
    assert {'SteadyStateMeshResults.xml', 'SteadyStateResults.xml',
            'pyHygroTherm.json'} <= set(members)
    provenance = json.loads(members['pyHygroTherm.json'])
    assert provenance['contract_version'] == 2001


def test_run_thmz_publishes_the_solve_members(tmp_path):
    target = tmp_path / 'published.thmz'
    published = Path(run_thmz(SLAB, output=target))
    members = solve(SLAB).members
    with zipfile.ZipFile(published) as archive:
        assert archive.namelist() == list(members)
        for name in archive.namelist():
            assert archive.read(name) == members[name]


def test_coded_exception_family(tmp_path):
    """Contract, mesh and unsupported errors carry stable code ranges."""
    with zipfile.ZipFile(SLAB) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    mesh = ET.fromstring(members['Mesh.xml'])
    mesh.find('Case/Elements/Element/node1').text = '9999'
    members['Mesh.xml'] = ET.tostring(mesh)
    bad_mesh = tmp_path / 'bad_mesh.thmz'
    with zipfile.ZipFile(bad_mesh, 'w') as archive:
        for name, content in members.items():
            archive.writestr(name, content)
    with pytest.raises(MeshError) as caught:
        solve(bad_mesh)
    assert caught.value.code == 'THMZ-E3001'
    assert isinstance(caught.value, ContractError)
    assert isinstance(caught.value, ValueError)
    assert str(caught.value).startswith('THMZ-E3001')

    model = ET.fromstring(members['Model.xml'])
    model.find('Properties/CalculationOptions/SimulateMoisture').text = 'true'
    members['Model.xml'] = ET.tostring(model)
    model_xml = ET.fromstring(members['Mesh.xml'])
    model_xml.find('Case/Elements/Element/node1').text = '0'
    members['Mesh.xml'] = ET.tostring(model_xml)
    wet = tmp_path / 'wet.thmz'
    with zipfile.ZipFile(wet, 'w') as archive:
        for name, content in members.items():
            archive.writestr(name, content)
    with pytest.raises(UnsupportedError) as caught:
        solve(wet)
    assert caught.value.code == 'THMZ-E5003'
    assert isinstance(caught.value, ValueError)


def test_kernel_failure_raises_solve_error_not_contract_error(
        monkeypatch, tmp_path):
    from pyhygrothermfem_fairyfly import _native

    def fail_solve(*args, **kwargs):
        raise RuntimeError('kernel did not converge')

    monkeypatch.setattr(_native, 'solve_steady', fail_solve)
    with pytest.raises(SolveError) as caught:
        solve(SLAB)
    assert caught.value.code is None
    assert isinstance(caught.value, THMZError)
    assert isinstance(caught.value, RuntimeError)
    assert isinstance(caught.value.__cause__, RuntimeError)


def test_cli_echoes_error_code_and_prints_no_result(tmp_path):
    """CLI failure text now carries the code but stays a ClickException."""
    from click.testing import CliRunner
    from pyhygrothermfem_fairyfly.cli import main

    source = tmp_path / 'missing_mesh.thmz'
    with zipfile.ZipFile(SLAB) as archive, zipfile.ZipFile(
            source, 'w') as target:
        for name in archive.namelist():
            if name != 'Mesh.xml':
                target.writestr(name, archive.read(name))
    result = CliRunner().invoke(main, ['run-thmz', str(source)])
    assert result.exit_code != 0
    assert 'THMZ-E1004' in result.stderr
    assert result.stdout == ''
    assert not source.with_name('missing_mesh_solved.thmz').exists()
