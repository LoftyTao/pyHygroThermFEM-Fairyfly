"""Inventory local notices without granting redistribution clearance."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import tarfile
import zipfile


NOTICE_SOURCES = {
    'HygroThermFEM/LICENSE': 'vendor/HygroThermFEM/LICENSE',
    'KeffCavity/LICENSE': 'vendor/KeffCavity/LICENSE',
    'LBNLCPPCommon/LICENSE': 'vendor/LBNLCPPCommon/LICENSE',
    'Windows-CalcEngine/License.txt': 'vendor/Windows-CalcEngine/License.txt',
    **{'Eigen/' + name: 'vendor/eigen/' + name for name in (
        'LICENSE', 'COPYING.README', 'COPYING.MPL2', 'COPYING.BSD',
        'COPYING.APACHE', 'COPYING.MINPACK')},
}
PYBIND11_LICENSE_SHA256 = (
    '83965b843b98f670d3a85bd041ed4b372c8ec50d7b4a5995a83ac697ba675dcb'
)
LOCAL_THMZ = {
    'tests/assets/slab.thmz', 'tests/assets/ticket06_local_cavity.thmz',
    'tests/assets/ticket07_series_wall.thmz',
    'tests/assets/thmz-contract/slab.thmz',
    'tests/assets/thmz-contract/transient-native-2.1.thmz',
    *{'tests/assets/ticket07_junction_{}.thmz'.format(count)
      for count in (80, 320, 1280, 5120)},
}


def _archive_members(path, kind):
    """Yield member names and bytes without extracting code or executables."""
    if kind == 'wheel':
        with zipfile.ZipFile(path) as archive:
            for member in archive.infolist():
                if not member.is_dir():
                    yield member.filename, archive.read(member)
    else:
        with tarfile.open(path, 'r:gz') as archive:
            for member in archive:
                if member.isfile():
                    with archive.extractfile(member) as stream:
                        yield member.name.split('/', 1)[1], stream.read()
                elif not member.isdir():
                    raise ValueError(
                        'non-regular sdist member: ' + member.name)


def audit_archive(path, source_root):
    """Compare archive notices and known exclusions with fixed local sources.

    Args:
        path: Local wheel or gzip source archive path.
        source_root: pyHygroTherm source directory containing native notices.

    Returns:
        Inventory and factual issues, never a legal clearance decision.
    """
    path, source_root = Path(path), Path(source_root)
    kind = 'wheel' if path.suffix == '.whl' else 'sdist'
    records = {}
    issues = []
    for name, data in _archive_members(path, kind):
        if name in records:
            issues.append('duplicate: ' + name)
        records[name] = {
            'path': name, 'size': len(data),
            'sha256': hashlib.sha256(data).hexdigest(),
        }
    expected = {
        'pyhygrothermfem_fairyfly/third_party_licenses/' + name:
            hashlib.sha256((source_root / source).read_bytes()).hexdigest()
        for name, source in NOTICE_SOURCES.items()
    }
    expected['pyhygrothermfem_fairyfly/third_party_licenses/pybind11/LICENSE'] = (
        PYBIND11_LICENSE_SHA256)
    expected['pyhygrothermfem_fairyfly/third_party_notices.txt'] = hashlib.sha256(
        (source_root / 'pyhygrothermfem_fairyfly/third_party_notices.txt'
         ).read_bytes()
    ).hexdigest()
    if kind == 'sdist':
        expected.update({source: hashlib.sha256(
            (source_root / source).read_bytes()).hexdigest()
            for source in NOTICE_SOURCES.values()})
        project_licenses = ['LICENSE']
    else:
        project_licenses = [name for name in records
                            if name.endswith('.dist-info/licenses/LICENSE')]
    if len(project_licenses) != 1:
        issues.append('missing or ambiguous: project LICENSE')
    else:
        expected[project_licenses[0]] = hashlib.sha256(
            (source_root / 'LICENSE').read_bytes()).hexdigest()
    for name, digest in expected.items():
        if name not in records:
            issues.append('missing: ' + name)
        elif records[name]['sha256'] != digest:
            issues.append('changed: ' + name)
    data_assets = []
    for name, record in records.items():
        normalized = '/' + name.replace('\\', '/').lower()
        suffix = Path(normalized).suffix
        if suffix in ('.thmz', '.csv', '.xml', '.json', '.dat', '.mtx'):
            data_assets.append(dict(record, scope='local-validation-only'))
        local_thmz = kind == 'sdist' and name in LOCAL_THMZ
        if local_thmz:
            digest = hashlib.sha256(
                (source_root / name).read_bytes()).hexdigest()
            if record['sha256'] != digest:
                issues.append('changed local fixture: ' + name)
        if (any(part in normalized for part in (
                '/vendor-local/', '/therm-reference/', '/simmetrix/'))
                or suffix in (
                    '.exe', '.dll', '.gh', '.ghuser', '.3dm',
                    '.lib', '.a', '.so', '.obj', '.pdf')
                or (suffix == '.thmz' and not local_thmz)
                or (suffix == '.pyd' and not (
                    kind == 'wheel'
                    and name.startswith('pyhygrothermfem_fairyfly/_native.')))):
            issues.append('excluded asset: ' + name)
    return {
        'path': str(path.resolve()),
        'kind': kind,
        'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'members': sorted(records.values(), key=lambda row: row['path']),
        'data_assets': sorted(data_assets, key=lambda row: row['path']),
        'issues': issues,
        'content_checks_passed': not issues,
        'redistribution_authorized': False,
    }


def audit_compile_commands(path):
    """Check registered translation units and includes for legacy build input.

    Args:
        path: CMake-generated compile_commands.json path.

    Returns:
        Registered source names and factual build-input issues.
        Registration alone does not prove a translation unit was linked.
    """
    path = Path(path)
    commands = json.loads(path.read_text(encoding='utf-8'))
    sources, issues = [], []
    for entry in commands:
        source = entry['file'].replace('\\', '/')
        sources.append(source)
        text = json.dumps(entry).replace('\\\\', '/').lower()
        if any(part in text for part in (
                '/vendor-local/', '/therm-reference/', '/simmetrix/')):
            issues.append('excluded build input: ' + source)
    for name in ('KeffC.cxx', 'KeffCavity.cxx'):
        suffix = '/vendor/KeffCavity/src/' + name
        if not any(source.endswith(suffix) for source in sources):
            issues.append('missing KeffCavity translation unit: ' + name)
    return {
        'path': str(path.resolve()),
        'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'registered_sources': sources,
        'issues': issues,
        'content_checks_passed': not issues,
        'redistribution_authorized': False,
    }


def inventory_environment():
    """Record installed metadata and notice files without inferring grants.

    Returns:
        Interpreter facts and all installed distributions, including raw
        conditional dependencies and embedded tool notices.
        These are evidence, not an automatic legal classification.
    """
    packages = []
    for distribution in importlib.metadata.distributions():
        metadata = distribution.metadata
        notices = []
        for member in distribution.files or ():
            if not member.name.lower().startswith((
                    'license', 'licence', 'copying', 'notice', 'copyright',
                    'authors')):
                continue
            path = Path(distribution.locate_file(member))
            notices.append({
                'member': str(member).replace('\\', '/'),
                'path': str(path.resolve()),
                'sha256': hashlib.sha256(path.read_bytes()).hexdigest()
                    if path.is_file() else None,
            })
        packages.append({
            'name': metadata['Name'], 'version': distribution.version,
            'license_expression': metadata.get('License-Expression'),
            'license_field': metadata.get('License'),
            'license_classifiers': [value for value in
                metadata.get_all('Classifier', [])
                if value.startswith('License')],
            'requires_dist': distribution.requires or [],
            'project_urls': metadata.get_all('Project-URL', []),
            'notices': sorted(notices, key=lambda row: row['member']),
        })
    return {
        'python': sys.version, 'executable': sys.executable,
        'isolated': bool(sys.flags.isolated),
        'packages': sorted(packages, key=lambda row: row['name'].lower()),
        'scope': 'metadata-evidence-only',
    }


def main():
    """Write a local evidence report; fail on package or build-input issues.

    Returns:
        Zero for successful content checks, one for failed checks.
        A successful exit never grants redistribution authorization.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path,
                        default=Path(__file__).resolve().parents[1])
    parser.add_argument('--archive', type=Path, action='append', default=[])
    parser.add_argument('--compile-commands', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = {
        'scope': 'local-build-license-evidence',
        'redistribution_authorized': False,
        'environment': inventory_environment(),
        'archives': [], 'build': None,
    }
    for path in args.archive:
        try:
            result = audit_archive(path, args.source_root)
        except (OSError, ValueError, KeyError, IndexError,
                zipfile.BadZipFile, tarfile.TarError) as error:
            result = {'path': str(path), 'issues': [str(error)],
                      'content_checks_passed': False,
                      'redistribution_authorized': False}
        report['archives'].append(result)
    if args.compile_commands:
        try:
            report['build'] = audit_compile_commands(args.compile_commands)
        except (OSError, ValueError, KeyError, TypeError) as error:
            report['build'] = {'issues': [str(error)],
                               'content_checks_passed': False}
    checks = report['archives'] + (
        [report['build']] if report['build'] else [])
    report['content_checks_passed'] = (
        all(check['content_checks_passed'] for check in checks)
        if checks else None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False)
                           + '\n', encoding='utf-8')
    print('Content checks:', report['content_checks_passed'], '|', args.output)
    return 1 if report['content_checks_passed'] is False else 0


if __name__ == '__main__':
    sys.exit(main())
