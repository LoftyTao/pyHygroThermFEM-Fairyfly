"""Transient result members: transient results/* and node moisture fields.

Contract-level tests for the current pyHygroTherm-owned result layout.
The partial THERM 8.1.30.0 archive observed by ticket 07 has different CSV
columns and path separators. These tests do not establish release-format
compatibility; that requires a qualified archive and a separate ticket.
"""
import csv
import io
import zipfile
import xml.etree.ElementTree as ET

import pytest

from pyhygrothermfem_fairyfly import ContractError
from pyhygrothermfem_fairyfly.transient_results import (
    TRANSIENT_RESULT_MEMBERS,
    TRANSIENT_RESULTS_DIR,
    TransientSeriesWriter,
    geometry_to_xml,
    read_geometry_xml,
    read_mesh_node_results,
    read_transient_series,
    save_transient_series,
    transient_result_members,
    write_mesh_node_results,
)


# --- Geometry.xml -----------------------------------------------------------

def _geometry():
    return {
        'materials': [{'id': 1, 'name': 'Foam', 'color': '0xFF0000'},
                      {'id': 2, 'name': 'Steel', 'color': '0x00FF00'}],
        'nodes': [{'id': 1, 'x': 0.0, 'y': 0.0},
                  {'id': 2, 'x': 0.05, 'y': 0.0},
                  {'id': 3, 'x': 0.05, 'y': 0.1},
                  {'id': 4, 'x': 0.0, 'y': 0.1}],
        'elements': [{'id': 1, 'node_ids': (1, 2, 3, 4), 'material_id': 1}],
        'material_polygons': [{'id': 1, 'material_id': 1,
                               'node_ids': (1, 2, 3, 4)}],
        'model_enclosure': (1, 2, 3, 4),
    }


def test_geometry_xml_roundtrip():
    """InputGeometryData mirrors the ModelGeometryForResults schema."""
    xml = geometry_to_xml(_geometry())
    root = ET.fromstring(xml)
    assert root.tag == 'InputGeometryData'
    assert root.findtext('Version') == '1'
    element = root.find('Elements/Element')
    assert [child.tag for child in element] == (
        ['ID'] + ['NodeID'] * 4 + ['MaterialID'])
    enclosure = [child.text for child in root.findall('ModelEnclosure/NodeID')]
    assert enclosure == ['1', '2', '3', '4']
    parsed = read_geometry_xml(xml)
    assert parsed == _geometry()


def test_geometry_xml_rejects_contract_violations():
    with pytest.raises(ValueError, match='THMZ-E20'):
        read_geometry_xml('<Wrong/>')
    bad = _geometry()
    bad['elements'] = [{'id': 1, 'node_ids': (1, 2, 3), 'material_id': 1}]
    with pytest.raises(ValueError, match='THMZ-E20'):
        geometry_to_xml(bad)
    bad = _geometry()
    bad['elements'][0]['node_ids'] = (1, 2, 3, 99)
    with pytest.raises(ValueError, match='THMZ-E30'):
        geometry_to_xml(bad)
    bad = _geometry()
    bad['nodes'] = bad['nodes'] + [dict(bad['nodes'][0])]
    with pytest.raises(ValueError, match='THMZ-E20'):
        geometry_to_xml(bad)


@pytest.mark.parametrize('node_path', (
    'ModelEnclosure/NodeID', 'MaterialPolygons/MaterialPolygon/NodeID'))
def test_geometry_xml_rejects_non_integer_reference_with_code(node_path):
    """Malformed references must use the result error family."""
    root = ET.fromstring(geometry_to_xml(_geometry()))
    root.find(node_path).text = 'not-an-integer'
    with pytest.raises(ContractError) as raised:
        read_geometry_xml(ET.tostring(root))
    assert raised.value.code == 'THMZ-E2000'
    assert node_path in str(raised.value)
    assert 'not-an-integer' in str(raised.value)


# --- transient results CSV --------------------------------------------------

def test_transient_series_writer_streams_and_roundtrips(tmp_path):
    """Rows append per timestep; memory stays O(one timestep)."""
    path = tmp_path / 'Temperatures.csv'
    writer = TransientSeriesWriter(path, entities=(1, 2, 3))
    writer.write_step(0.0, (20.0, 20.0, 20.0))
    writer.write_step(3600.0, (19.9, 19.8, 19.7))
    writer.close()
    lines = path.read_text().splitlines()
    assert len(lines) == 3  # header + two steps
    header = lines[0].split(',')
    assert header[0].lower().startswith('time')
    assert [cell for cell in header[1:]] == ['1', '2', '3']
    series = read_transient_series(path.read_bytes())
    assert series['times'] == [0.0, 3600.0]
    assert series['entities'] == (1, 2, 3)
    assert series['values'][1] == [19.9, 19.8, 19.7]
    # append another step on a reopened writer
    writer = TransientSeriesWriter(tmp_path / 'Humidities.csv',
                                   entities=(1, 2, 3))
    writer.write_step(0.0, (0.5, 0.5, 0.5))
    writer.write_step(1800.0, (0.6, 0.6, 0.6))
    writer.close()
    series = read_transient_series(
        (tmp_path / 'Humidities.csv').read_bytes())
    assert series['times'] == [0.0, 1800.0]


def test_transient_series_write_member_bytes():
    """save_transient_series returns the CSV bytes used in the archive."""
    blob = save_transient_series((10, 20), [(0.0, (1.0, 2.0)),
                                            (60.0, (3.0, 4.0))])
    series = read_transient_series(blob)
    assert series['entities'] == (10, 20)
    assert series['values'] == [[1.0, 2.0], [3.0, 4.0]]
    assert series['times'] == [0.0, 60.0]


def test_transient_series_rejects_contract_violations(tmp_path):
    with pytest.raises(ValueError, match='THMZ-E20'):
        TransientSeriesWriter(tmp_path / 'x.csv', entities=(1, 2))\
            .write_step(0.0, (1.0,))
    path = tmp_path / 'bad.csv'
    path.write_text('Time (s),1,2\n0,1,2\n-1,3,4\n')
    with pytest.raises(ValueError, match='THMZ-E20'):
        read_transient_series(path.read_bytes())
    path.write_text('Time (s),1,2\n0,1,abc\n')
    with pytest.raises(ValueError, match='THMZ-E20'):
        read_transient_series(path.read_bytes())


def test_transient_series_rejects_invalid_utf8_with_code():
    """Invalid result bytes must not leak a UnicodeDecodeError."""
    with pytest.raises(ContractError) as raised:
        read_transient_series(b'Time (s),1\n0,\xff\n')
    assert raised.value.code == 'THMZ-E2000'
    assert 'UTF-8' in str(raised.value)


@pytest.mark.parametrize('content', (
    b'', b'Other,1\n0,20\n', b'Time (s),not-an-id\n0,20\n',
    b'Time (s),1\n0,20,21\n', b'Time (s),1\nnot-a-time,20\n',
    b'Time (s),1\nnan,20\n', b'Time (s),1\n0,inf\n',
    b'Time (s),1\n1,20\n0,21\n'))
def test_transient_series_errors_have_machine_readable_code(content):
    """Current-format invalid inputs retain E2000 on the exception object."""
    with pytest.raises(ContractError) as raised:
        read_transient_series(content)
    assert raised.value.code == 'THMZ-E2000'


def test_transient_series_accepts_utf8_bom_and_equal_times():
    """BOM removal and equal times preserve the existing read contract."""
    parsed = read_transient_series(
        b'\xef\xbb\xbfTime (s),11,21\r\n0,20,21\r\n0,19,20\r\n')
    assert parsed['entities'] == (11, 21)
    assert parsed['times'] == [0.0, 0.0]
    assert parsed['values'] == [[20.0, 21.0], [19.0, 20.0]]


def test_transient_series_flushes_caller_stream_without_closing(tmp_path):
    """A caller-owned file publishes each step and stays open after exit."""
    path = tmp_path / 'external-stream.csv'
    with path.open('w', encoding='utf-8', newline='') as stream:
        with TransientSeriesWriter(stream, (1, 2)) as writer:
            writer.write_step(0.0, (20.0, 21.0))
            assert path.read_bytes() == b'Time (s),1,2\r\n0,20,21\r\n'
            writer.write_step(60.0, (19.0, 20.0))
            assert read_transient_series(path.read_bytes())['times'] == (
                [0.0, 60.0])
        assert not stream.closed
        stream.write('120,18,19\r\n')
        stream.flush()
        assert read_transient_series(path.read_bytes())['times'] == (
            [0.0, 60.0, 120.0])


def test_result_member_names_match_current_contract():
    """Slash-separated paths are the owned layout, not release evidence."""
    assert TRANSIENT_RESULTS_DIR == 'transient results'
    assert TRANSIENT_RESULT_MEMBERS == {
        'geometry': 'transient results/Geometry.xml',
        'temperatures': 'transient results/Temperatures.csv',
        'humidities': 'transient results/Humidities.csv',
        'heat_flux': 'transient results/HeatFlux.csv',
        'heat_flux_edges': 'transient results/HeatFluxEdges.csv',
        'water_content': 'transient results/WaterContent.csv',
        'water_flux': 'transient results/WaterFlux.csv',
        'water_flux_edges': 'transient results/WaterFluxEdges.csv',
    }


# --- SteadyStateMeshResults node moisture fields ----------------------------

MESH_RESULTS = (
    '<MeshResults><Version>1</Version><Case><ResultsType>U-factor'
    '</ResultsType><Nodes>'
    '<Node><Index>1</Index><Temperature>-17.5</Temperature>'
    '<X-flux>-14.1</X-flux><Y-flux>-0.43</Y-flux></Node>'
    '<Node><Index>2</Index><Temperature>10.0</Temperature>'
    '<X-flux>1.0</X-flux><Y-flux>0.0</Y-flux>'
    '<Humidity>0.55</Humidity><WaterContent>24.5</WaterContent></Node>'
    '</Nodes></Case></MeshResults>')


def test_read_mesh_node_results_returns_moisture_fields():
    """Own XML path surfaces fields the pylibraryfemtherm binding drops."""
    nodes = read_mesh_node_results(MESH_RESULTS)
    assert len(nodes) == 2
    assert nodes[0] == {'index': 1, 'temperature': -17.5, 'x_flux': -14.1,
                        'y_flux': -0.43, 'humidity': None,
                        'water_content': None}
    assert nodes[1]['humidity'] == 0.55
    assert nodes[1]['water_content'] == 24.5
    assert nodes[1]['index'] == 2


def test_write_mesh_node_results_includes_moisture_fields():
    """The own writer keeps humidity and waterContent round-trippable."""
    xml = write_mesh_node_results([
        {'index': 1, 'temperature': -17.5, 'x_flux': -14.1, 'y_flux': -0.43},
        {'index': 2, 'temperature': 10.0, 'x_flux': 1.0, 'y_flux': 0.0,
         'humidity': 0.55, 'water_content': 24.5}])
    parsed = read_mesh_node_results(xml)
    assert parsed[1]['humidity'] == 0.55
    assert parsed[1]['water_content'] == 24.5
    root = ET.fromstring(xml)
    assert root.tag == 'MeshResults'
    node = root.findall('Case/Nodes/Node')[1]
    assert [child.tag for child in node] == [
        'Index', 'Temperature', 'X-flux', 'Y-flux', 'Humidity',
        'WaterContent']


def test_read_mesh_node_results_validates():
    with pytest.raises(ValueError, match='THMZ-E20'):
        read_mesh_node_results('<MeshResults><Case><Nodes><Node>'
                               '<Index>1</Index></Node></Nodes></Case>'
                               '</MeshResults>')
    with pytest.raises(ValueError, match='THMZ-E20'):
        read_mesh_node_results('<MeshResults><Case/><Case/></MeshResults>')


@pytest.mark.parametrize('fields', (
    {}, {'humidity': None, 'water_content': None},
    {'humidity': 0.0, 'water_content': 0.0},
    {'humidity': 0.55}, {'water_content': 24.5},
    {'humidity': 1.02, 'water_content': 24.5}))
def test_node_moisture_roundtrip_preserves_absence_zero_and_finite_values(
        fields):
    """XML preserves finite values without fabricating or clipping them."""
    record = {'index': 11, 'temperature': 19.0,
              'x_flux': -1.0, 'y_flux': 2.0}
    record.update(fields)
    parsed = read_mesh_node_results(write_mesh_node_results([record]))[0]
    assert parsed['humidity'] == fields.get('humidity')
    assert parsed['water_content'] == fields.get('water_content')
    assert parsed['index'] == 11


@pytest.mark.parametrize('field', ('temperature', 'x_flux', 'y_flux',
                                  'humidity', 'water_content'))
def test_node_writer_rejects_nonfinite_values_with_code(field):
    """Node scalars reject non-finite data rather than replacing it."""
    record = {'index': 1, 'temperature': 20.0, 'x_flux': 0.0, 'y_flux': 0.0,
              'humidity': 0.5, 'water_content': 20.0}
    record[field] = float('nan')
    with pytest.raises(ContractError) as raised:
        write_mesh_node_results([record])
    assert raised.value.code == 'THMZ-E2000'


# --- archive-level integration ----------------------------------------------

def test_transient_result_members_pack_into_archive(tmp_path):
    """All seven CSVs plus Geometry.xml land under transient results/."""
    geometry = _geometry()
    archive = tmp_path / 'case.thmz'
    entities = [n['id'] for n in geometry['nodes']]
    steps = [(0.0, [20.0] * 4), (3600.0, [19.0] * 4)]
    members = transient_result_members(geometry, {
        key: (entities, steps)
        for key in ('temperatures', 'humidities', 'heat_flux',
                    'heat_flux_edges', 'water_content', 'water_flux',
                    'water_flux_edges')})
    with zipfile.ZipFile(archive, 'w') as bundle:
        for name, data in members.items():
            bundle.writestr(name, data)
    with zipfile.ZipFile(archive) as bundle:
        assert set(bundle.namelist()) == set(TRANSIENT_RESULT_MEMBERS.values())
        parsed = read_geometry_xml(bundle.read(
            TRANSIENT_RESULT_MEMBERS['geometry']))
        assert parsed['nodes'][1]['x'] == 0.05
        for key in ('temperatures', 'humidities', 'heat_flux',
                    'heat_flux_edges', 'water_content', 'water_flux',
                    'water_flux_edges'):
            series = read_transient_series(bundle.read(
                TRANSIENT_RESULT_MEMBERS[key]))
            assert series['times'] == [0.0, 3600.0]
            assert series['entities'] == (1, 2, 3, 4)
            assert series['values'] == [[20.0] * 4, [19.0] * 4]


def test_unknown_result_member_is_rejected():
    """Unknown result keys are rejected rather than silently discarded."""
    with pytest.raises(ContractError) as raised:
        transient_result_members(_geometry(), {'unknown': ((1,), [(0, [1])])})
    assert raised.value.code == 'THMZ-E2000'
