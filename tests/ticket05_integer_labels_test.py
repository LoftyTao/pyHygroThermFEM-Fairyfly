"""Public result writers must preserve integer labels without truncation."""
import io

import pytest

from pyhygrothermfem_fairyfly import ContractError
from pyhygrothermfem_fairyfly.transient_results import (
    TransientSeriesWriter, geometry_to_xml, read_geometry_xml,
    read_mesh_node_results, read_transient_series, save_transient_series,
    transient_result_members, write_mesh_node_results)


BAD_LABELS = (1.5, True, False, None, float('nan'), float('inf'),
              float('-inf'), '1.5', 'not-an-id')
LABEL_FIELDS = (
    ('materials', 'id', 'Materials/Material/ID'),
    ('nodes', 'id', 'Nodes/Node/ID'),
    ('elements', 'id', 'Elements/Element/ID'),
    ('elements', 'node_ids', 'Elements/Element/NodeID'),
    ('elements', 'material_id', 'Elements/Element/MaterialID'),
    ('material_polygons', 'id', 'MaterialPolygons/MaterialPolygon/ID'),
    ('material_polygons', 'material_id',
     'MaterialPolygons/MaterialPolygon/MaterialID'),
    ('material_polygons', 'node_ids',
     'MaterialPolygons/MaterialPolygon/NodeID'),
    ('model_enclosure', None, 'ModelEnclosure/NodeID'),
)


def _geometry():
    return {
        'materials': [{'id': 1, 'name': 'Foam', 'color': '0xFF0000'}],
        'nodes': [{'id': 1, 'x': 0.0, 'y': 0.0},
                  {'id': 2, 'x': 1.0, 'y': 0.0},
                  {'id': 3, 'x': 1.0, 'y': 1.0},
                  {'id': 4, 'x': 0.0, 'y': 1.0}],
        'elements': [{'id': 1, 'node_ids': (1, 2, 3, 4), 'material_id': 1}],
        'material_polygons': [{'id': 1, 'node_ids': (1, 2, 3, 4),
                               'material_id': 1}],
        'model_enclosure': (1, 2, 3, 4),
    }


def _replace_label(geometry, group, key, value):
    if key == 'node_ids':
        geometry[group][0][key] = (value, 2, 3, 4)
    elif key is None:
        geometry[group] = (value, 2, 3, 4)
    else:
        geometry[group][0][key] = value


@pytest.mark.parametrize('group,key,field', LABEL_FIELDS)
@pytest.mark.parametrize('value', BAD_LABELS)
def test_geometry_writer_rejects_lossy_or_invalid_labels(
        group, key, field, value):
    """Every geometry label entry reports the member, field and raw value."""
    geometry = _geometry()
    _replace_label(geometry, group, key, value)
    with pytest.raises(ContractError) as raised:
        geometry_to_xml(geometry)
    assert raised.value.code == 'THMZ-E2000'
    assert 'transient results/Geometry.xml' in str(raised.value)
    assert field in str(raised.value)
    assert repr(value) in str(raised.value)


@pytest.mark.parametrize('interface', ('stream', 'save', 'members'))
@pytest.mark.parametrize('value', BAD_LABELS)
def test_csv_writer_interfaces_reject_invalid_entity_labels(interface, value):
    """Public streaming, byte and archive writers share the integer rule."""
    with pytest.raises(ContractError) as raised:
        if interface == 'stream':
            TransientSeriesWriter(io.StringIO(), (value,))
        elif interface == 'save':
            save_transient_series((value,), [(0.0, [20.0])])
        else:
            transient_result_members(_geometry(), {
                'temperatures': ((value,), [(0.0, [20.0])])})
    assert raised.value.code == 'THMZ-E2000'
    assert 'transient results csv' in str(raised.value)
    assert 'entity ID' in str(raised.value)
    assert repr(value) in str(raised.value)


@pytest.mark.parametrize('value', BAD_LABELS)
def test_node_writer_rejects_invalid_index_labels(value):
    """Node Index cannot silently lose a fractional part or accept bool."""
    record = {'index': value, 'temperature': 20.0,
              'x_flux': 0.0, 'y_flux': 0.0}
    with pytest.raises(ContractError) as raised:
        write_mesh_node_results([record])
    assert raised.value.code == 'THMZ-E2000'
    assert 'SteadyStateMeshResults.xml' in str(raised.value)
    assert 'Index' in str(raised.value)
    assert repr(value) in str(raised.value)


@pytest.mark.parametrize('group,key,field', LABEL_FIELDS)
@pytest.mark.parametrize('value', (1, 1.0))
def test_geometry_writer_preserves_lossless_integer_labels(
        group, key, field, value):
    """Integral floats remain accepted without changing serialized labels."""
    geometry = _geometry()
    _replace_label(geometry, group, key, value)
    assert geometry_to_xml(geometry) == geometry_to_xml(_geometry())
    assert read_geometry_xml(geometry_to_xml(geometry)) == _geometry()


@pytest.mark.parametrize('value,expected', ((1, 1), (1.0, 1), ('1', 1),
                                         ('+7', 7), ('0007', 7), (-3, -3),
                                         (0, 0)))
def test_scalar_writer_labels_keep_lossless_legacy_inputs(value, expected):
    """Whole integer text, zero and negative labels are not newly forbidden."""
    parsed = read_transient_series(save_transient_series(
        (value,), [(0.0, [20.0])]))
    assert parsed['entities'] == (expected,)
    node = read_mesh_node_results(write_mesh_node_results([
        {'index': value, 'temperature': 20.0,
         'x_flux': 0.0, 'y_flux': 0.0}]))[0]
    assert node['index'] == expected


def test_geometry_labels_do_not_gain_a_positive_only_constraint():
    """Valid signed/zero integer labels retain their existing roundtrip."""
    geometry = _geometry()
    labels = (-4, -3, -2, -1)
    geometry['materials'][0]['id'] = 0
    for node, label in zip(geometry['nodes'], labels):
        node['id'] = label
    geometry['elements'][0].update(id=-9, node_ids=labels, material_id=0)
    geometry['material_polygons'][0].update(
        id=0, node_ids=labels, material_id=0)
    geometry['model_enclosure'] = labels
    assert read_geometry_xml(geometry_to_xml(geometry)) == geometry
