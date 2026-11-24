"""Verify the license and asset boundaries of local build packages."""

import hashlib
import io
import json
from pathlib import Path
import tarfile
import zipfile

import pytest


ROOT = Path(__file__).resolve().parents[1]


def make_wheel(path, changed=None, omitted=None, extra=None):
    """Create a package fixture from the committed notice texts.

    Args:
        path: Output ZIP path.
        changed: Member whose notice is replaced with invalid text.
        omitted: Member removed from the package.
        extra: Additional member name to include.

    Returns:
        Path to the package fixture.
    """
    members = {
        'local.dist-info/licenses/LICENSE': (ROOT / 'LICENSE').read_bytes(),
        'pyhygrothermfem_fairyfly/third_party_notices.txt': (
            ROOT / 'pyhygrothermfem_fairyfly/third_party_notices.txt'
        ).read_bytes(),
    }
    for notice in (ROOT / 'pyhygrothermfem_fairyfly/third_party_licenses').glob('*/*'):
        members[notice.relative_to(ROOT).as_posix()] = notice.read_bytes()
    if changed:
        members[changed] = b'not the upstream license'
    if omitted:
        del members[omitted]
    if extra:
        members[extra] = b'local-only asset'
    with zipfile.ZipFile(path, 'w') as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    return path


def test_packaged_pybind11_notice_matches_pinned_license():
    """The header dependency keeps the exact pybind11 3.1.0 notice."""
    notice = ROOT / 'pyhygrothermfem_fairyfly/third_party_licenses/pybind11/LICENSE'
    assert hashlib.sha256(notice.read_bytes()).hexdigest() == (
        '83965b843b98f670d3a85bd041ed4b372c8ec50d7b4a5995a83ac697ba675dcb'
    )


def test_archive_notice_bytes_match_sources(tmp_path):
    """A package must retain both dependency and project notice texts."""
    from license_inventory import audit_archive

    archive = make_wheel(tmp_path / 'local.whl')
    result = audit_archive(archive, ROOT)
    assert result['issues'] == []
    assert result['redistribution_authorized'] is False

    changed = make_wheel(
        tmp_path / 'changed.whl',
        changed='pyhygrothermfem_fairyfly/third_party_licenses/HygroThermFEM/LICENSE')
    result = audit_archive(changed, ROOT)
    assert ('changed: pyhygrothermfem_fairyfly/third_party_licenses/HygroThermFEM/LICENSE'
            in result['issues'])


@pytest.mark.parametrize('extra', [
    'vendor-local/FairyCavity/src/legacy.cxx',
    'therm-reference/materials.xml',
    'Simmetrix/mesher.bin',
    'pyhygrothermfem_fairyfly/THERM8.exe',
    'pyhygrothermfem_fairyfly/fortran.dll',
    'pyhygrothermfem_fairyfly/materials.thmz',
])
def test_archive_rejects_local_only_runtime_assets(tmp_path, extra):
    """Known external runtimes and unreviewed wheel data stay outside."""
    from license_inventory import audit_archive

    result = audit_archive(
        make_wheel(tmp_path / 'local.whl', extra=extra), ROOT)
    assert 'excluded asset: ' + extra in result['issues']


def test_sdist_keeps_original_notices_and_local_fixture_provenance(tmp_path):
    """Original notices and local data do not imply release clearance."""
    from license_inventory import audit_archive

    wheel = make_wheel(tmp_path / 'local.whl')
    members = {}
    with zipfile.ZipFile(wheel) as archive:
        for name in archive.namelist():
            target = 'LICENSE' if name.startswith('local.dist-info') else name
            members[target] = archive.read(name)
    for notice in (ROOT / 'pyhygrothermfem_fairyfly/third_party_licenses').glob('*/*'):
        if notice.parent.name == 'pybind11':
            continue
        component = 'eigen' if notice.parent.name == 'Eigen' else (
            notice.parent.name)
        source = 'vendor/' + component + '/' + notice.name
        members[source] = (ROOT / source).read_bytes()
    members['tests/assets/slab.thmz'] = (
        ROOT / 'tests/assets/slab.thmz').read_bytes()
    path = tmp_path / 'local.tar.gz'
    with tarfile.open(path, 'w:gz') as archive:
        for name, data in members.items():
            info = tarfile.TarInfo('local-1.0/' + name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    result = audit_archive(path, ROOT)
    assert result['issues'] == []
    assert result['data_assets'][0]['path'] == 'tests/assets/slab.thmz'
    assert result['data_assets'][0]['scope'] == 'local-validation-only'
    assert result['redistribution_authorized'] is False


def test_compile_inventory_rejects_legacy_include_paths(tmp_path):
    """Build inputs must resolve to KeffCavity, not vendor-local or therm-reference."""
    from license_inventory import audit_compile_commands

    commands = [
        {'file': 'D:/src/vendor/KeffCavity/src/' + name,
         'command': 'cl /I D:/src/vendor/KeffCavity/include ' + name}
        for name in ('KeffC.cxx', 'KeffCavity.cxx')
    ]
    path = tmp_path / 'compile_commands.json'
    path.write_text(json.dumps(commands), encoding='utf-8')
    assert audit_compile_commands(path)['issues'] == []
    commands[0]['command'] += ' /I D:\\src\\vendor-local\\FairyCavity\\include'
    path.write_text(json.dumps(commands), encoding='utf-8')
    result = audit_compile_commands(path)
    assert result['issues']
    assert result['content_checks_passed'] is False
