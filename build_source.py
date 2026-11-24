"""Verify build inputs against Git objects, including in Git-free source packages.

This private build helper uses only the standard library. The proof establishes
local object/content consistency, not a signature or an upstream certification.
"""

import argparse
import base64
import configparser
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import tempfile


_PROOF_NAME = '_source_freeze.json'
_ROOT_FILES = (
    'CMakeLists.txt', 'MANIFEST.in', 'build_source.py', 'pyproject.toml',
    'requirements.txt', 'setup.py',
)
_VENDOR_ROOTS = (
    'vendor/HygroThermFEM', 'vendor/KeffCavity',
    'vendor/Windows-CalcEngine', 'vendor/LBNLCPPCommon', 'vendor/eigen',
)
_SOURCE_ROOTS = ('bindings', 'pyhygrothermfem_fairyfly') + _VENDOR_ROOTS
_CONFIG_INPUTS = (
    'vendor/HygroThermFEM/doc/HygroThermFEM.md.in',
    'vendor/KeffCavity/doc/KeffCavity.md.in',
)
_ROOT_METADATA = {
    '.gitattributes', '.gitignore', '.gitmodules', '.releaserc.json', '.coverage',
    'LICENSE', 'README.md', 'README.zh-CN.md', 'CONTRIBUTING.md', 'PKG-INFO',
    _PROOF_NAME,
}
# These are non-compiling directories at known project roots, never directories
# named "doc" or "build" underneath a compiling src/include tree. In particular,
# HTFEM and KeffCavity configure_file rewrite doc/DependencyChart.md on configure.
_NON_INPUT_ROOTS = (
    'docs', 'patches', 'tests', '.github', '.pytest_cache', 'build', 'dist',
    'pyhygrothermfem_fairyfly.egg-info', 'pyHygroThermFEM_Fairyfly.egg-info',
) + tuple(
    root + '/' + name
    for root in _VENDOR_ROOTS
    for name in ('.github', 'doc', 'docs', 'build')
)
_MODES = {'040000', '100644', '100755', '120000', '160000'}


class SourceError(RuntimeError):
    """The source proof or a build input is missing or inconsistent."""


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=True).encode('utf-8')


def _object_sha1(kind, data):
    header = kind.encode('ascii') + b' ' + str(len(data)).encode('ascii') + b'\0'
    return hashlib.sha1(header + data).hexdigest()


def _hex(value, length, context):
    if not isinstance(value, str) or re.fullmatch('[0-9a-f]{%d}' % length,
                                                value) is None:
        raise SourceError(
            'Invalid %s digest: received %r; expected %d lowercase hex '
            'characters (all-zero syntax is permitted)' % (context, value, length))
    return value


def _shape(value, keys, context):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise SourceError(
            'Unknown or incomplete %s structure: received %r; required '
            'fields %r' % (context, value, tuple(keys)))


def _schema(value, context):
    if type(value) is not int or value != 1:
        raise SourceError('Unknown %s schema: received %r; expected 1' %
                          (context, value))


def _relative(value, allow_root=False):
    if allow_root and value == '.':
        return value
    if not isinstance(value, str) or not value or '\\' in value or '\0' in value:
        raise SourceError('Invalid source path: %r' % value)
    parts = value.split('/')
    if (any(part in ('', '.', '..') or ':' in part
            or part.endswith((' ', '.')) or part.casefold() == '.git'
            for part in parts)
            or PurePosixPath(value).as_posix() != value):
        raise SourceError('Unsafe source path: %r' % value)
    return value


def _join(repository, path):
    return path if repository == '.' else repository + '/' + path


def _excluded(path):
    if path in _CONFIG_INPUTS or any(name.startswith(path + '/') for name in _CONFIG_INPUTS):
        return False
    return any(path == root or path.startswith(root + '/')
               for root in _NON_INPUT_ROOTS)


def _is_input(path):
    if path in _ROOT_FILES:
        return True
    if path in _ROOT_METADATA or path.endswith(('.pyc', '.pyo')):
        return False
    if _excluded(path):
        return False
    if path.startswith('pyhygrothermfem_fairyfly/'):
        if (path.endswith(('.pyc', '.pyo'))
                or path == 'pyhygrothermfem_fairyfly/_build_source.json'):
            return False
        name = PurePosixPath(path).name
        if (PurePosixPath(path).parent == PurePosixPath('pyhygrothermfem_fairyfly')
                and name.startswith('_native') and name.endswith(('.pyd', '.so'))):
            return False
    # Any other root/control/source file is an input, irrespective of suffix.
    # Only the explicit metadata, docs, and generated-output exceptions above
    # are outside this closure; optional CMake includes cannot add unseen files.
    return True


def _git(repo, *args):
    # Git environment selectors and replacement objects must not redirect the
    # requested repository/objects. Version environment variables are unrelated.
    env = {key: value for key, value in os.environ.items()
           if not key.upper().startswith('GIT_')}
    result = subprocess.run(
        ['git', '--no-optional-locks', '--no-replace-objects', '-C', str(repo),
         *args], stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
    if result.returncode:
        raise SourceError('Git source verification failed in %s: %s' % (
            repo, result.stderr.decode('utf-8', errors='replace').strip()))
    return result.stdout


def _git_root(repo):
    if not (repo / '.git').exists():
        raise SourceError('Missing Git repository for source closure: %s' % repo)
    actual = Path(_git(repo, 'rev-parse', '--show-toplevel').decode('utf-8').strip())
    if actual.resolve() != repo.resolve():
        raise SourceError('Gitlink is not a separate repository: %s' % repo)
    if _git(repo, 'rev-parse', '--show-object-format').strip() != b'sha1':
        raise SourceError('Only SHA-1 Git object proofs are supported')


def _commit_tree(data):
    first_line = data.split(b'\n', 1)[0]
    if not first_line.startswith(b'tree ') or b'\n\n' not in data:
        raise SourceError('Invalid Git commit object')
    try:
        return _hex(first_line[5:].decode('ascii'), 40, 'commit tree')
    except UnicodeDecodeError as error:
        raise SourceError('Invalid commit tree encoding') from error


def _git_proof(repo, commit, require_heads=False):
    repositories = []

    def visit(directory, revision, relative):
        _git_root(directory)
        _hex(revision, 40, 'source commit')
        if require_heads:
            head = _git(directory, 'rev-parse', '--verify', 'HEAD').decode().strip()
            if head != revision:
                raise SourceError('Gitlink checkout differs from parent tree: %s' % relative)
            staged = _git(directory, 'diff', '--cached', '--name-only', '-z', revision)
            for name in staged.split(b'\0'):
                if name:
                    path = _join(relative, name.decode('utf-8'))
                    if path in _SOURCE_ROOTS or _is_input(path):
                        raise SourceError('Staged build input differs from commit: %s' % path)
        data = _git(directory, 'cat-file', 'commit', revision)
        entries = []
        rows = _git(directory, 'ls-tree', '-r', '-t', '--full-tree', '-z', revision)
        for row in rows.split(b'\0'):
            if not row:
                continue
            fields, name = row.split(b'\t', 1)
            mode, kind, digest = fields.decode('ascii').split()
            expected_kind = 'tree' if mode == '040000' else (
                'commit' if mode == '160000' else 'blob')
            if kind != expected_kind:
                raise SourceError('Invalid Git tree entry type')
            entries.append({'path': name.decode('utf-8'), 'mode': mode, 'sha1': digest})
        repositories.append({
            'path': relative, 'commit': revision, 'tree': _commit_tree(data),
            'commit_object_base64': base64.b64encode(data).decode('ascii'),
            'entries': entries,
        })
        for entry in entries:
            if entry['mode'] == '160000':
                child = _relative(entry['path'])
                visit(directory / child, entry['sha1'], _join(relative, child))

    visit(repo, commit, '.')
    return {'schema_version': 1, 'repositories': repositories, 'files': {}}


def _verify_objects(proof):
    _shape(proof, ('schema_version', 'repositories', 'files'),
           '_source_freeze.json: source proof')
    _schema(proof['schema_version'], '_source_freeze.json: source proof')
    if not isinstance(proof['repositories'], list) or not proof['repositories']:
        raise SourceError('Missing repository closure in _source_freeze.json: '
                          'repositories received %r; expected nonempty list' %
                          proof['repositories'])
    if not isinstance(proof['files'], dict):
        raise SourceError('Missing build input records in _source_freeze.json: '
                          'files received %r; expected object' % proof['files'])
    repositories = {}
    folded_repositories = set()
    for index, repository in enumerate(proof['repositories']):
        location = '_source_freeze.json: repositories[%d]' % index
        _shape(repository, ('path', 'commit', 'tree', 'commit_object_base64', 'entries'),
               location)
        name = _relative(repository['path'], allow_root=True)
        if name.casefold() in folded_repositories:
            raise SourceError('Duplicate repository path in %s.path: received %r; '
                              'expected unique repository path' % (location, name))
        folded_repositories.add(name.casefold())
        commit = _hex(repository['commit'], 40, location + '.commit')
        tree = _hex(repository['tree'], 40, location + '.tree')
        try:
            data = base64.b64decode(repository['commit_object_base64'], validate=True)
        except (ValueError, TypeError) as error:
            raise SourceError('Invalid commit object encoding in %s: received %r; '
                              'expected base64 commit bytes' %
                              (location, repository['commit_object_base64'])) from error
        if _object_sha1('commit', data) != commit or _commit_tree(data) != tree:
            raise SourceError('Commit object digest/tree mismatch in %s: '
                              'received commit %r, tree %r; expected matching '
                              'Git commit object and tree' % (location, commit, tree))
        if not isinstance(repository['entries'], list):
            raise SourceError('Missing flat Git tree entries in %s.entries: '
                              'received %r; expected list' %
                              (location, repository['entries']))
        entries = {}
        folded_entries = set()
        for entry_index, entry in enumerate(repository['entries']):
            entry_location = location + '.entries[%d]' % entry_index
            _shape(entry, ('path', 'mode', 'sha1'), entry_location)
            path = _relative(entry['path'])
            if path.casefold() in folded_entries:
                raise SourceError('Duplicate Git tree path in %s.path: received %r; '
                                  'expected unique path' % (entry_location, path))
            folded_entries.add(path.casefold())
            if not isinstance(entry['mode'], str) or entry['mode'] not in _MODES:
                raise SourceError('Unsupported Git tree mode in %s.mode: received '
                                  '%r; expected one of %r' %
                                  (entry_location, entry['mode'], sorted(_MODES)))
            _hex(entry['sha1'], 40, entry_location + '.sha1')
            entries[path] = dict(entry)
        trees = {'': tree}
        children = {'': []}
        for path, entry in entries.items():
            if entry['mode'] == '040000':
                trees[path] = entry['sha1']
                children[path] = []
        for path, entry in entries.items():
            parent = path.rpartition('/')[0]
            if parent not in trees:
                raise SourceError('Missing parent tree for %s' % path)
            children[parent].append(entry)
        for directory in sorted(trees, key=lambda path: path.count('/') + bool(path),
                                reverse=True):
            rows = sorted(children[directory], key=lambda entry: (
                entry['path'].rsplit('/', 1)[-1].encode('utf-8') +
                (b'/' if entry['mode'] == '040000' else b'\0')))
            contents = b''.join(
                entry['mode'].lstrip('0').encode('ascii') + b' ' +
                entry['path'].rsplit('/', 1)[-1].encode('utf-8') + b'\0' +
                bytes.fromhex(entry['sha1']) for entry in rows)
            if _object_sha1('tree', contents) != trees[directory]:
                raise SourceError('Git tree digest mismatch: %s/%s' % (name, directory))
        repositories[name] = {
            'path': name, 'commit': commit, 'tree': tree,
            'commit_object_base64': base64.b64encode(data).decode('ascii'),
            'entries': [entries[path] for path in sorted(entries)],
        }
    if '.' not in repositories:
        raise SourceError('Missing root repository proof')
    reachable = {'.'}
    for name, repository in repositories.items():
        for entry in repository['entries']:
            if entry['mode'] == '160000':
                child = _join(name, entry['path'])
                if child not in repositories or repositories[child]['commit'] != entry['sha1']:
                    raise SourceError('Incomplete or mismatched Gitlink proof: %s' % child)
                reachable.add(child)
    if reachable != set(repositories):
        raise SourceError('Repository proof contains an unlinked repository')
    root_entries = {entry['path']: entry for entry in repositories['.']['entries']}
    kernel = root_entries.get('vendor/HygroThermFEM')
    if kernel is None or kernel['mode'] not in ('040000', '160000'):
        raise SourceError('Missing parent-tree HygroThermFEM source entry')
    inputs = {}
    for name, repository in repositories.items():
        for entry in repository['entries']:
            path = _join(name, entry['path'])
            if entry['mode'] in ('040000', '160000') or not _is_input(path):
                continue
            if entry['mode'] == '120000':
                raise SourceError('Symlink build inputs are unsupported: %s' % path)
            if path in inputs:
                raise SourceError('Overlapping repository build inputs: %s' % path)
            inputs[path] = entry['sha1']
    if not set(_ROOT_FILES).issubset(inputs):
        raise SourceError('Commit does not declare all required root build inputs')
    for root in _SOURCE_ROOTS:
        if not any(path.startswith(root + '/') for path in inputs):
            raise SourceError('Commit does not declare build input root: %s' % root)
    normalized = {
        'schema_version': 1,
        'repositories': [repositories[name] for name in sorted(repositories)],
        'files': proof['files'],
    }
    return normalized, inputs, kernel['sha1']


def _is_link(path):
    return path.is_symlink() or path.is_junction()


def _source_bytes(root, name):
    path = root
    for part in _relative(name).split('/'):
        path = path / part
        if _is_link(path):
            raise SourceError('Linked build input is unsupported: %s' % name)
    if not path.is_file():
        raise SourceError('Missing build input: %s' % name)
    return path.read_bytes()


def _blob_record(data, expected, path):
    normalization = 'none'
    if _object_sha1('blob', data) != expected:
        # Do not decode or re-encode legacy source text. Permit only CRLF
        # conversion, no binary controls or bare CR, and an exact Git digest.
        normalized = data.replace(b'\r\n', b'\n')
        if (b'\0' in data or b'\r\n' not in data
                or b'\r' in normalized
                or re.search(br'[\x01-\x08\x0e-\x1f\x7f]', data)
                or _object_sha1('blob', normalized) != expected):
            raise SourceError('Build input blob mismatch: %s' % path)
        normalization = 'crlf-to-lf'
    return {'blob_sha1': expected, 'raw_sha256': hashlib.sha256(data).hexdigest(),
            'normalization': normalization}


def _version_metadata(path):
    """Recognize only setuptools' generated, non-compiling egg_info settings."""
    if _is_link(path):
        raise SourceError('Linked build metadata is unsupported: %s' % path)
    parser = configparser.RawConfigParser()
    try:
        parser.read_string(path.read_text(encoding='utf-8'))
    except (configparser.Error, UnicodeError):
        return False
    return (not parser.defaults() and parser.sections() == ['egg_info']
            and set(parser['egg_info']).issubset({'tag_build', 'tag_date'}))


def _input_inventory(root, inputs, git_metadata):
    inventory = set()

    def walk_error(error):
        raise SourceError('Cannot enumerate build inputs: %s' % error) from error

    for name in _SOURCE_ROOTS:
        directory = root / name
        if not directory.is_dir() or _is_link(directory):
            raise SourceError('Missing or linked build input root: %s' % name)
    for current, directories, files in os.walk(root, onerror=walk_error):
        parent = Path(current)
        kept = []
        for child in directories:
            path = (parent / child).relative_to(root).as_posix()
            if path.casefold() in git_metadata or _excluded(path):
                continue
            if _is_link(parent / child):
                raise SourceError('Linked build input directory: %s' % path)
            _relative(path)
            kept.append(child)
        directories[:] = kept
        for child in files:
            path = (parent / child).relative_to(root).as_posix()
            if path.casefold() in git_metadata or not _is_input(path):
                continue
            if (path == 'setup.cfg' and path not in inputs
                    and _version_metadata(parent / child)):
                continue
            _relative(path)
            inventory.add(path)
    return inventory


def _verify_source(root, proof, freezing=False, git_checkout=False):
    proof, inputs, kernel = _verify_objects(proof)
    if not freezing and set(proof['files']) != set(inputs):
        raise SourceError('Incomplete or undeclared build input records')
    # Only authenticated repository roots have Git metadata, and only a live
    # Git checkout may omit those markers. Git-free packages must omit them.
    git_metadata = {
        ('.git' if item['path'] == '.' else item['path'] + '/.git').casefold()
        for item in proof['repositories']
    } if git_checkout else set()
    inventory = _input_inventory(root, inputs, git_metadata)
    if inventory != set(inputs):
        missing = sorted(set(inputs) - inventory)
        extra = sorted(inventory - set(inputs))
        raise SourceError('Build input inventory mismatch; missing=%s; undeclared=%s' %
                          (missing, extra))
    records = {}
    for path, blob in sorted(inputs.items()):
        record = _blob_record(_source_bytes(root, path), blob, path)
        if not freezing:
            supplied = proof['files'][path]
            location = '_source_freeze.json: files[%r]' % path
            _shape(supplied, ('blob_sha1', 'raw_sha256', 'normalization'),
                   location)
            _hex(supplied['blob_sha1'], 40, location + '.blob_sha1')
            _hex(supplied['raw_sha256'], 64, location + '.raw_sha256')
            if supplied != record:
                raise SourceError('Build input byte record mismatch: %s; '
                                  'received %r; actual %r' %
                                  (location, supplied, record))
        records[path] = record
    proof['files'] = records
    identity = {
        'schema_version': 1,
        'source_commit': next(repository['commit'] for repository in proof['repositories']
                              if repository['path'] == '.'),
        'kernel_commit': kernel,
        'source_closure_sha256': hashlib.sha256(_canonical(proof)).hexdigest(),
        'python_sha256': {
            path[len('pyhygrothermfem_fairyfly/'):]: record['raw_sha256']
            for path, record in records.items()
            if (path.startswith('pyhygrothermfem_fairyfly/')
                and path.endswith('.py'))
        },
    }
    return proof, identity


def _unique_object(pairs, context='_source_freeze.json'):
    result = {}
    for key, value in pairs:
        if key in result:
            raise SourceError('Duplicate metadata key in %s: %s; received %r'
                              % (context, key, value))
        result[key] = value
    return result


def _read_json(path):
    if not path.is_file() or _is_link(path):
        raise SourceError('Missing or linked source metadata: %s' % path)
    received = path.read_bytes()
    try:
        received = received.decode('utf-8')
        return json.loads(received,
                          object_pairs_hook=lambda pairs: _unique_object(
                              pairs, str(path)))
    except (ValueError, UnicodeError) as error:
        raise SourceError('Invalid source metadata: %s; received %r; reason: %s' %
                          (path, received, error)) from error


def _write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.read_bytes() == data:
        return
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data)
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def freeze_source(repo: Path, commit: str, destination: Path) -> dict:
    """Write a Git-object proof after verifying an already exported source tree.

    Args:
        repo: Real Git checkout with all recursive Gitlink objects available.
        commit: Full lower-case SHA-1 of the root commit to export.
        destination: Existing export of that exact recursive source closure.

    Returns:
        The verified private build identity, also returned by prepare_source.

    Raises:
        SourceError: An object, Gitlink, or actual build input is inconsistent.
    """
    repo, destination = Path(repo).resolve(), Path(destination).resolve()
    proof, identity = _verify_source(destination, _git_proof(repo, commit), freezing=True)
    _write(destination / _PROOF_NAME, _canonical(proof) + b'\n')
    return identity


def prepare_source(root: Path) -> dict:
    """Verify a clean Git checkout or a self-contained frozen source package.

    Args:
        root: Source root containing .git or the mandatory frozen object proof.

    Returns:
        The private identity of the verified actual build inputs.

    Raises:
        SourceError: The proof, Git closure, or actual build inputs are invalid.
    """
    return _prepared_source(Path(root))[1]


def _prepared_source(root):
    """Return the validated manifest as well as identity for source packaging."""
    root = Path(root).resolve()
    marker = root / '.git'
    if marker.exists() or marker.is_symlink():
        _git_root(root)
        commit = _git(root, 'rev-parse', '--verify', 'HEAD').decode().strip()
        proof = _git_proof(root, commit, require_heads=True)
        return _verify_source(root, proof, freezing=True, git_checkout=True)
    return _verify_source(root, _read_json(root / _PROOF_NAME))


def _verify_identity(identity, context='build-source-identity.json'):
    _shape(identity, ('schema_version', 'source_commit', 'kernel_commit',
                      'source_closure_sha256', 'python_sha256'),
           context + ': build identity')
    _schema(identity['schema_version'], context + ': build identity')
    _hex(identity['source_commit'], 40, context + ': source_commit')
    _hex(identity['kernel_commit'], 40, context + ': kernel_commit')
    _hex(identity['source_closure_sha256'], 64,
         context + ': source_closure_sha256')
    if not isinstance(identity['python_sha256'], dict) or not identity['python_sha256']:
        raise SourceError('Missing package Python byte records in %s: '
                          'python_sha256 received %r; expected nonempty object' %
                          (context, identity['python_sha256']))
    for path, digest in identity['python_sha256'].items():
        if not _relative(path).endswith('.py'):
            raise SourceError('Invalid package Python path: %s' % path)
        _hex(digest, 64, context + ': python_sha256/' + path)


def finalize_package(output_dir: Path, identity_path: Path, native_path: Path) -> dict:
    """Write companion metadata beside the native extension after compilation.

    Args:
        output_dir: Extension output directory, normally the package in build_lib.
        identity_path: Canonical identity file generated during configuration.
        native_path: Actual compiled _native Windows extension in output_dir.

    Returns:
        The metadata written to the companion _build_source.json file.

    Raises:
        SourceError: The identity or compiled output is missing or invalid.
    """
    output_dir = Path(output_dir).resolve()
    native_path = Path(native_path).absolute()
    if (native_path.parent.resolve() != output_dir or _is_link(native_path)
            or not native_path.is_file() or not native_path.name.startswith('_native')
            or native_path.suffix != '.pyd'):
        raise SourceError('Missing or misplaced compiled native extension: %s' % native_path)
    identity = _read_json(Path(identity_path))
    _verify_identity(identity, str(identity_path))
    metadata = {
        'schema_version': 1, 'identity': identity,
        'native_sha256': hashlib.sha256(native_path.read_bytes()).hexdigest(),
    }
    _write(output_dir / '_build_source.json', _canonical(metadata) + b'\n')
    return metadata


def _configure(root, header, identity_file):
    root, header, identity_file = map(Path.resolve, (root, header, identity_file))
    if header == identity_file:
        raise SourceError('Header and identity output paths must differ')
    for output in (header, identity_file):
        if output.is_relative_to(root) and _is_input(output.relative_to(root).as_posix()):
            raise SourceError('Generated metadata cannot overwrite build inputs')
    identity = prepare_source(root)
    contents = _canonical(identity)
    literal = json.dumps(contents.decode('utf-8'), ensure_ascii=True)
    source = ('// Generated by build_source.py; do not edit.\n'
              '#pragma once\n\n'
              'namespace pyhygrotherm\n{\n'
              'inline constexpr char kBuildSourceIdentity[] =\n    ' + literal + ';\n}\n')
    _write(header, source.encode('utf-8'))
    _write(identity_file, contents + b'\n')


def main():
    """Generate checked native identity files from command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    configure = commands.add_parser('configure')
    configure.add_argument('--root', type=Path, required=True)
    configure.add_argument('--header', type=Path, required=True)
    configure.add_argument('--identity-file', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'configure':
        _configure(args.root, args.header, args.identity_file)


if __name__ == '__main__':
    main()
