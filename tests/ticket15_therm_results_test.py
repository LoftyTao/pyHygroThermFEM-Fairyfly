"""THERM result layout tests at the member reader/writer boundary."""
from pathlib import Path
import io
import os
import zipfile
import xml.etree.ElementTree as ET

import pytest

from pyhygrothermfem_fairyfly import ContractError, MeshError
from pyhygrothermfem_fairyfly import transient_results as results


_ARCHIVE_DIR = (
    Path(os.environ['THERM81_ARCHIVE_DIR'])
    if os.environ.get('THERM81_ARCHIVE_DIR')
    else None
)
ARCHIVES = (
    _ARCHIVE_DIR / 'therm-once-cli' / 'case.thmz'
    if _ARCHIVE_DIR else Path('__optional__/therm-once-cli/case.thmz'),
    _ARCHIVE_DIR / 'Stucco Wall - Moisture.thmz'
    if _ARCHIVE_DIR else Path('__optional__/Stucco Wall - Moisture.thmz'),
)


def _synthetic_therm_bundle():
    """Return a minimal valid 10-step THERM transient result bundle."""
    geometry = {
        'materials': [{'id': 1, 'name': 'Brick', 'color': '#C0C0C0'}],
        'nodes': [
            {'id': 1, 'x': 0.0, 'y': 0.0},
            {'id': 2, 'x': 0.1, 'y': 0.0},
            {'id': 3, 'x': 0.1, 'y': 0.1},
            {'id': 4, 'x': 0.0, 'y': 0.1},
        ],
        'elements': [{
            'id': 1, 'node_ids': (1, 2, 3, 4), 'material_id': 1,
        }],
        'material_polygons': [{
            'id': 1, 'material_id': 1, 'node_ids': (1, 2, 3, 4),
        }],
        'model_enclosure': (1, 2, 3, 4),
    }
    frames = [(
        -1, 0.0, [20.0] * 4, [0.5] * 4, [10.0] * 4, [],
        [(0.0, 0.0)] * 4, [(0.0, 0.0)] * 4, [], [], [], [], 0.0, 0.0,
    )]
    for step in range(1, 11):
        frames.append((
            step - 1, 360.0 * step, [20.0 + step * 0.1] * 4, [0.5] * 4,
            [10.0] * 4, [], [(1.0, -0.5)] * 4, [(0.1, 0.0)] * 4,
            [], [], [], [], 0.001, 0.001,
        ))
    members = results.therm_transient_result_members(
        geometry, frames, vector_convention='geometry-xy-signed')
    return {name.replace('\\', '/'): data for name, data in members.items()}


@pytest.mark.parametrize('path,count', tuple(zip(ARCHIVES, (138, 30))))
def test_real_therm_scalar_and_geometry_read(path, count):
    """Real archives, not synthetic fixtures, fix the node-domain columns."""
    if not path.is_file():
        pytest.skip(
            'Optional external THERM 8.1 reference archive not present; '
            'set THERM81_ARCHIVE_DIR to run.')
    with zipfile.ZipFile(path) as archive:
        members = {i.filename.replace('\\', '/'): archive.read(i)
                   for i in archive.infolist()}
    geometry = results.read_geometry_xml(
        members['transient results/Geometry.xml'])
    parsed = results.read_therm_transient_series(
        members['transient results/Temperatures.csv'])
    assert len(geometry['nodes']) == count
    assert parsed['entities'] == tuple(range(1, count + 1))
    assert parsed['steps'] == list(range(1, 11 if count == 138 else 101))
    assert parsed['times'] is None
    assert parsed['kind'] == 'scalar'
    assert parsed['values'][0][0] == (20.05358835 if count == 138 else
                                     18.36308329)


@pytest.mark.parametrize('path,count,steps', (
    (ARCHIVES[0], 138, 10), (ARCHIVES[1], 30, 100)))
def test_real_therm_vector_and_edge_tables(path, count, steps):
    if not path.is_file():
        pytest.skip(
            'Optional external THERM 8.1 reference archive not present; '
            'set THERM81_ARCHIVE_DIR to run.')
    with zipfile.ZipFile(path) as archive:
        for field in ('HeatFlux', 'WaterFlux'):
            parsed = results.read_therm_transient_series(archive.read(
                'transient results/' + field + '.csv'))
            assert parsed['kind'] == 'vector'
            assert parsed['entities'] == tuple(range(1, count + 1))
            assert len(parsed['values']) == steps
            assert parsed['times'] is None
            assert parsed['errors'] is None
            assert len(parsed['values'][0]) == count
            assert len(parsed['values'][0][0]) == 2
        heat = results.read_therm_transient_series(archive.read(
            'transient results/HeatFlux.csv'))
        assert heat['values'][0][0][0] == (
            1.613611374 if count == 138 else -103.7805368)
        for field in ('HeatFluxEdges', 'WaterFluxEdges'):
            parsed = results.read_therm_transient_series(archive.read(
                'transient results/' + field + '.csv'))
            assert parsed['kind'] == 'edges'
            assert parsed['times'] == [360.0 * i for i in range(1, steps + 1)]
            assert parsed['durations'] == [360.0] * steps
            assert parsed['entities'] == ()
            assert parsed['values'] == [[] for _ in range(steps)]


@pytest.mark.parametrize('kind,entities,values,error,expected', (
    ('scalar', (1, 2), (20, 21), 0.25,
     'timestep #,error tolerance,1,2\n1,0.25,20,21\n'),
    ('vector', (1, 2), ((-3, 4), (5, -6)), None,
     '1-x,1-y,2-x,2-y\n-3,4,5,-6\n'),
    ('edges', (), (), None,
     'Total time,Timestep time,\nsec,sec,\n'
     '360.000000,360.000000,\nTotal,,\n')))
def test_therm_stream_writer(kind, entities, values, error, expected):
    stream = io.StringIO()
    with results.ThermTransientSeriesWriter(
            stream, entities, kind,
            vector_convention='geometry-xy-signed') as writer:
        writer.write_step(360, values, error)
        if kind == 'edges':
            assert stream.getvalue().endswith('360.000000,360.000000,\n')
        else:
            assert stream.getvalue() == expected
    assert not stream.closed
    assert stream.getvalue() == expected
    parsed = results.read_therm_transient_series(stream.getvalue())
    assert parsed['steps'] == [1]
    assert parsed['kind'] == kind


def test_therm_vector_writer_rejects_unknown_coordinate_convention():
    with pytest.raises(ContractError, match='THMZ-E2000'):
        results.ThermTransientSeriesWriter(io.StringIO(), (1,), 'vector')


def _members(path):
    with zipfile.ZipFile(path) as archive:
        return {i.filename.replace('\\', '/'): archive.read(i)
                for i in archive.infolist()}


@pytest.mark.parametrize('path', ARCHIVES)
def test_all_real_members_roundtrip_each_column_and_value(path, tmp_path):
    """All seven real CSVs retain every value, error and row on replay."""
    if not path.is_file():
        pytest.skip(
            'Optional external THERM 8.1 reference archive not present; '
            'set THERM81_ARCHIVE_DIR to run.')
    original = results.read_therm_transient_results(_members(path))
    series = original['series']
    frames = []
    for i, time in enumerate(series['heat_flux_edges']['times']):
        frames.append((
            i, time, series['temperatures']['values'][i],
            series['humidities']['values'][i],
            series['water_content']['values'][i], [],
            series['heat_flux']['values'][i],
            series['water_flux']['values'][i], [], [], [], [],
            series['temperatures']['errors'][i],
            series['humidities']['errors'][i]))
    written = results.therm_transient_result_members(
        original['geometry'], frames,
        vector_convention='geometry-xy-signed')
    assert len(written) == 8
    assert all(name.startswith('transient results\\') for name in written)
    geometry = ET.fromstring(written['transient results\\Geometry.xml'])
    assert geometry.find('Version') is not None
    assert geometry.findtext('Version') == ''
    assert results.read_geometry_xml(ET.tostring(geometry)) == original['geometry']
    reread = results.read_therm_transient_results(written)
    for key, reference in series.items():
        for field in ('labels', 'entities', 'steps', 'values', 'times', 'errors'):
            assert reread['series'][key][field] == reference[field]
    assert written['transient results\\HeatFluxEdges.csv'] == \
        written['transient results\\WaterFluxEdges.csv']
    with zipfile.ZipFile(tmp_path / 'therm.thmz', 'w') as archive:
        archive.writestr('timestep input files/keep.xml', b'unchanged')
        results.write_therm_transient_members(archive, written)
    with zipfile.ZipFile(tmp_path / 'therm.thmz') as archive:
        assert 'timestep input files/keep.xml' in [
            i.orig_filename for i in archive.infolist()]
        raw_names = [i.orig_filename for i in archive.infolist()
                     if i.orig_filename.startswith('transient results')]
        assert set(raw_names) == set(written)
        restored = {i.orig_filename: archive.read(i)
                    for i in archive.infolist()}
        assert results.read_therm_transient_results(restored) == reread


@pytest.mark.parametrize('content', (
    b'', b'\xff', b'timestep #,error tolerance,2\n1,0,20\n',
    b'timestep #,error tolerance,1\n0,0,20\n',
    b'timestep #,error tolerance,1\n2,0,20\n',
    b'timestep #,error tolerance,1\n1,0\n',
    b'timestep #,error tolerance,1\n1,nan,20\n',
    b'timestep #,error tolerance,1\n1,0,inf\n',
    b'1-x\n1\n', b'1-x,2-y\n1,2\n', b'1-x,1-y\n1\n',
    b'1-x,1-y\n1,nan\n', b'1-x,1-y\n"1,2\n',
    b'Total time,Timestep time,\nsec,sec,\n360,360,\n',
    b'Total time,Timestep time,\nsec,sec,\n0,0,\nTotal,,\n',
    b'Total time,Timestep time,\nsec,sec,\n360,60,\nTotal,,\n',
    b'Total time,Timestep time,\nsec,sec,\n360,360,5\nTotal,,\n',
))
def test_therm_reader_contract_errors(content):
    with pytest.raises(ContractError) as raised:
        results.read_therm_transient_series(content)
    assert raised.value.code == 'THMZ-E2000'


@pytest.mark.parametrize('field', (
    'MaterialPolygons/MaterialPolygon/NodeID', 'ModelEnclosure/NodeID'))
def test_bundle_reader_keeps_raw_backslash_member_in_diagnostics(field):
    """Dangling NodeID diagnostics name the raw backslash member."""
    members = _synthetic_therm_bundle()
    geometry_name = 'transient results\\Geometry.xml'
    members[geometry_name] = members['transient results/Geometry.xml']
    del members['transient results/Geometry.xml']
    root = ET.fromstring(members[geometry_name])
    root.find(field).text = '999999'
    members[geometry_name] = ET.tostring(root)
    with pytest.raises(MeshError) as raised:
        results.read_therm_transient_results(members)
    assert raised.value.code == 'THMZ-E3000'
    assert geometry_name in str(raised.value)
    assert 'transient results/Geometry.xml' not in str(raised.value)
    assert field in str(raised.value)


@pytest.mark.parametrize('kind,entities,values,time,error', (
    ('scalar', (1,), (20,), 0, None),
    ('scalar', (1,), (20,), float('nan'), None),
    ('scalar', (1,), (20,), 360, float('inf')),
    ('scalar', (1,), (float('nan'),), 360, None),
    ('scalar', (1,), (), 360, None),
    ('vector', (1,), (5,), 360, None),
    ('vector', (1,), ((1,),), 360, None),
    ('vector', (1,), ((1, 2, 3),), 360, None),
    ('vector', (1,), ((1, float('inf')),), 360, None),
    ('edges', (), (5,), 360, None),
))
def test_therm_writer_contract_errors(kind, entities, values, time, error):
    stream = io.StringIO()
    with results.ThermTransientSeriesWriter(
            stream, entities, kind,
            vector_convention='geometry-xy-signed') as writer:
        before = stream.getvalue()
        with pytest.raises(ContractError) as raised:
            writer.write_step(time, values, error)
        assert raised.value.code == 'THMZ-E2000'
        assert stream.getvalue() == before


def test_initial_frame_is_omitted_and_owned_layout_unchanged():
    geometry = results.read_geometry_xml(
        _synthetic_therm_bundle()['transient results/Geometry.xml'])
    count = len(geometry['nodes'])
    initial = (-1, 0, [20] * count, [0.5] * count, [2] * count, [],
               [(0, 0)] * count, [(0, 0)] * count, [], [], [], [], 0, 0)
    completed = (0, 60, [21] * count, [0.6] * count, [3] * count, [],
                 [(-1, 2)] * count, [(0, 0)] * count, [], [], [], [],
                 None, None)
    written = results.therm_transient_result_members(
        geometry, (initial, completed),
        vector_convention='geometry-xy-signed')
    read = results.read_therm_transient_results(written)
    assert read['series']['temperatures']['steps'] == [1]
    assert read['series']['temperatures']['errors'] == [0]
    assert read['series']['humidities']['errors'] == [0]
    assert read['series']['water_content']['errors'] == [0]
    assert read['series']['heat_flux']['values'][0][0] == (-1, 2)
    assert ET.fromstring(results.geometry_to_xml(geometry)).findtext('Version') == '1'
    owned = results.save_transient_series((1,), ((0, (20,)), (60, (21,))))
    assert owned == b'Time (s),1\r\n0,20\r\n60,21\r\n'


@pytest.mark.parametrize('mutation,code', (
    ('duplicate', 'THMZ-E1003'), ('missing', 'THMZ-E1004'),
    ('count', 'THMZ-E2000'), ('kind', 'THMZ-E2000'),
    ('time', 'THMZ-E2000'), ('geometry', 'THMZ-E2000')))
def test_bundle_cross_member_contract_errors(mutation, code):
    members = _synthetic_therm_bundle()
    if mutation == 'duplicate':
        members['transient results\\Geometry.xml'] = members[
            'transient results/Geometry.xml']
    elif mutation == 'missing':
        del members['transient results/WaterFlux.csv']
    elif mutation == 'count':
        lines = members['transient results/HeatFlux.csv'].splitlines(True)
        members['transient results/HeatFlux.csv'] = b''.join(lines[:-1])
    elif mutation == 'kind':
        members['transient results/HeatFlux.csv'] = members[
            'transient results/Temperatures.csv']
    elif mutation == 'time':
        members['transient results/WaterFluxEdges.csv'] = (
            b'Total time,Timestep time,\nsec,sec,\n' +
            b''.join(('{0:.6f},60.000000,\n'.format(i * 60)).encode()
                     for i in range(1, 11)) + b'Total,,\n')
    else:
        root = ET.fromstring(members['transient results/Geometry.xml'])
        root.find('Nodes/Node/ID').text = '999'
        members['transient results/Geometry.xml'] = ET.tostring(root)
    with pytest.raises(ContractError) as raised:
        results.read_therm_transient_results(members)
    if mutation == 'geometry':
        assert raised.value.code in ('THMZ-E2000', 'THMZ-E3000')
    else:
        assert raised.value.code == code


@pytest.mark.parametrize('kind,entities', (
    ('scalar', ()), ('scalar', (2,)), ('scalar', (1, 1)),
    ('scalar', (1.5,)), ('scalar', (True,)),
    ('edges', (1,)), ('unknown', (1,))))
def test_therm_writer_rejects_bad_ids_and_kinds(kind, entities):
    with pytest.raises(ContractError, match='THMZ-E2000'):
        results.ThermTransientSeriesWriter(io.StringIO(), entities, kind)


def test_therm_stream_close_and_step_order(tmp_path):
    path = tmp_path / 'edges.csv'
    writer = results.ThermTransientSeriesWriter(path, (), 'edges')
    writer.write_step(60)
    assert path.read_bytes().endswith(b'60.000000,60.000000,\n')
    for time in (60, 30):
        with pytest.raises(ContractError, match='THMZ-E2000'):
            writer.write_step(time)
    writer.write_step(180)
    writer.close()
    writer.close()
    assert path.read_bytes().count(b'Total,,') == 1
    assert results.read_therm_transient_series(path.read_bytes())[
        'durations'] == [60, 120]
    with pytest.raises(ContractError, match='THMZ-E2000'):
        writer.write_step(240)


@pytest.mark.parametrize('frames', (
    (), ((0,),),
    ((1, 60, [20], [0.5], [2], [], [(0, 0)], [(0, 0)], [], [], [], [],
      0, 0),),
    ((-1, 60, [20], [0.5], [2], [], [(0, 0)], [(0, 0)], [], [], [], [],
      0, 0),),
))
def test_bundle_writer_rejects_empty_or_bad_frames(frames):
    geometry = results.read_geometry_xml(
        _synthetic_therm_bundle()['transient results/Geometry.xml'])
    with pytest.raises(ContractError, match='THMZ-E2000'):
        results.therm_transient_result_members(
            geometry, frames, vector_convention='geometry-xy-signed')


def test_geometry_version_is_required_but_may_be_empty():
    with pytest.raises(ContractError, match='THMZ-E2000'):
        results.read_geometry_xml('<InputGeometryData/>')
    assert results.read_geometry_xml('<InputGeometryData><Version/>'
                                     '</InputGeometryData>')['nodes'] == []


def test_edge_writer_rejects_unrepresentable_microsecond_steps():
    stream = io.StringIO()
    writer = results.ThermTransientSeriesWriter(stream, (), 'edges')
    before = stream.getvalue()
    with pytest.raises(ContractError, match='THMZ-E2000'):
        writer.write_step(1e-8)
    assert stream.getvalue() == before


def test_zip_writer_rejects_duplicate_results_before_writing():
    original = results.read_therm_transient_results(_synthetic_therm_bundle())
    geometry = original['geometry']
    n = len(geometry['nodes'])
    frame = (0, 360, [20] * n, [0.5] * n, [2] * n, [],
             [(0, 0)] * n, [(0, 0)] * n, [], [], [], [], 0, 0)
    members = results.therm_transient_result_members(
        geometry, [frame], vector_convention='geometry-xy-signed')
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w') as archive:
        archive.writestr('transient results/Temperatures.csv', b'old')
        count = len(archive.infolist())
        with pytest.raises(ContractError, match='THMZ-E1003'):
            results.write_therm_transient_members(archive, members)
        assert len(archive.infolist()) == count
