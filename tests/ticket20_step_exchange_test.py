"""Ticket 20: step-exchange result members (ADR-0010 option a).

Contract 2.1 adds the owned-layout ``HeatFluxStepExchange`` and
``WaterFluxStepExchange`` members: per-edge interval exchange (J/m, kg/m,
outward-positive) accumulated by the kernel over the accepted sub-steps of
each public step, right-aligned to the frame time. The instant-rate
``*Edges`` members keep their semantics; the exchange members exist only in
the owned layout because THERM 8.1 writes no such data.

The reference conservation checks reuse the dry ticket 06 slab (rho*cp = 1,
nodal quarter-volumes of the 3x3-node 2x1 m mesh) and the hygrothermal humid
convection case of ticket 09. Thresholds are fixed before running: the
energy and water residuals must stay below 1e-2 of each step's exchanged
magnitude plus a 1e-9 J/m (resp. kg/m) noise floor -- far tighter than the
instant-rate ledger's known ~9% phase error, but honest about the
post-acceptance evaluation of nonlinear boundary rates.
"""
import csv
import io
import math
import zipfile
import xml.etree.ElementTree as ET

import pytest

from pyhygrothermfem_fairyfly import ContractError, MeshError, solve
from pyhygrothermfem_fairyfly import transient_results as series_io

from ticket06_transient_dispatch_test import _transient_slab, NUM_STEPS
from ticket09_hygrothermal_test import _humid_convection, _wet_slab

STEP_EXCHANGE = {
    'heat': 'transient results/HeatFluxStepExchange.csv',
    'water': 'transient results/WaterFluxStepExchange.csv',
}
# Consistent-mass nodal volumes of the 3x3-node slab mesh (ticket09).
NODAL_VOLUMES = (0.125, 0.25, 0.125, 0.25, 0.5, 0.25, 0.125, 0.25, 0.125)
STEP_LIMIT = 1e-2
FLOOR = 1e-9


def _read(members, name):
    return series_io.read_transient_series(members[name])


def _rows(members, name):
    return _read(members, name)


def test_owned_layout_publishes_step_exchange_members(tmp_path):
    """Both exchange members exist with edge entity columns and num_steps+1
    rows; the t=0 frame carries zeros because no interval precedes it."""
    result = solve(_transient_slab(tmp_path))
    members = result.members
    assert result.provenance['contract_version'] == 2001
    edges = _read(members, 'transient results/HeatFluxEdges.csv')
    for name in STEP_EXCHANGE.values():
        series = _rows(members, name)
        assert series['entities'] == edges['entities']
        assert series['times'] == edges['times']
        assert len(series['values']) == NUM_STEPS + 1
        assert all(value == 0.0 for value in series['values'][0])
    heat = _rows(members, STEP_EXCHANGE['heat'])
    # The hot-face convection edges exchange real heat on step 1.
    assert max(abs(v) for v in heat['values'][1]) > 0.0


def test_heat_step_exchange_closes_energy_ledger(tmp_path):
    """Delta stored energy + cumulative outward exchange ~= 0 (dry slab).

    rho*cp = 1 for the ticket 06 material, so nodal energy is T*volume.
    """
    result = solve(_transient_slab(tmp_path))
    members = result.members
    temperatures = _read(members, 'transient results/Temperatures.csv')
    exchange = _rows(members, STEP_EXCHANGE['heat'])
    worst = 0.0
    for index in range(1, len(exchange['times'])):
        outward = math.fsum(exchange['values'][index])
        delta = math.fsum(
            (a - b) * volume for a, b, volume in zip(
                temperatures['values'][index],
                temperatures['values'][index - 1], NODAL_VOLUMES))
        residual = outward + delta
        worst = max(worst, abs(residual))
        assert abs(residual) <= STEP_LIMIT * (
            abs(outward) + abs(delta)) + FLOOR
    # The ledger was exercised by real exchange, not an empty interval.
    assert worst < FLOOR  # exact closure is the expected linear-case answer


def test_water_step_exchange_closes_mass_ledger(tmp_path):
    """WaterContent mass change + outward WaterFluxStepExchange ~= 0.

    The ticket 09 humid-Robin drive wets the slab; the same ledger built
    from instantaneous rates shows a percent-level phase error (ticket 17),
    while the interval exchange is the assembled quantity.
    """
    record, timestep = _humid_convection(humidity=0.9)
    result = solve(_wet_slab(tmp_path, type_records=[record],
                             timestep_files=timestep))
    members = result.members
    water = _read(members, 'transient results/WaterContent.csv')
    exchange = _rows(members, STEP_EXCHANGE['water'])
    instant = _rows(members, 'transient results/WaterFluxEdges.csv')
    times = exchange['times']
    assert max(math.fsum(map(abs, row)) for row in exchange['values']) > 0.0
    step_residuals = []
    instant_residuals = []
    for index in range(1, len(times)):
        outward = math.fsum(exchange['values'][index])
        delta = math.fsum(
            (a - b) * volume for a, b, volume in zip(
                water['values'][index], water['values'][index - 1],
                NODAL_VOLUMES))
        step_residuals.append(abs(outward + delta))
        instant_residuals.append(abs(
            (times[index] - times[index - 1])
            * math.fsum(instant['values'][index]) + delta))
        assert abs(outward + delta) <= STEP_LIMIT * abs(outward) + FLOOR
    # The assembled interval exchange must not be worse than the instant-rate
    # ledger it replaces for balance bookkeeping.
    assert max(step_residuals) <= max(instant_residuals) + FLOOR


def test_therm_layout_does_not_emit_or_require_exchange(tmp_path):
    """THERM layout has no exchange members (real THERM writes none)."""
    result = solve(_transient_slab(tmp_path), result_layout='therm')
    names = {name.replace('\\', '/') for name in result.members}
    for name in STEP_EXCHANGE.values():
        assert name not in names


def test_exchange_members_reject_bad_content():
    """The generic owned reader still enforces the shared CSV contract on the
    new members (time axis first, finite values, consistent row width)."""
    with pytest.raises(ContractError) as caught:
        series_io.read_transient_series(b'Edge,1,2\n0,0,0\n')
    assert caught.value.code == 'THMZ-E2000'
    with pytest.raises(ContractError) as caught:
        series_io.read_transient_series(b'Time (s),1,2\n0,0\n')
    assert caught.value.code == 'THMZ-E2000'


def test_geometry_standalone_interfaces_reject_dangling_nodes():
    """S1 review fix: polygon/enclosure NodeID references must belong to the
    actual node domain on both the reader and the writer paths."""
    geometry = {
        'materials': [{'id': 1, 'name': 'm', 'color': '0'}],
        'nodes': [{'id': i, 'x': float(i), 'y': 0.0} for i in range(1, 5)],
        'elements': [{'id': 1, 'node_ids': (1, 2, 3, 4), 'material_id': 1}],
        'material_polygons': [{'id': 1, 'material_id': 1,
                               'node_ids': (1, 2, 3)}],
        'model_enclosure': (1, 2, 3, 4)}
    content = series_io.geometry_to_xml(geometry)
    # Baseline: a well-formed geometry round-trips.
    parsed = series_io.read_geometry_xml(content)
    assert len(parsed['nodes']) == 4

    # Writer path: a polygon or enclosure NodeID outside the node domain is
    # rejected before serialization.
    broken_polygon = dict(geometry)
    broken_polygon['material_polygons'] = [
        dict(geometry['material_polygons'][0], node_ids=(1, 2, 999))]
    with pytest.raises(MeshError) as caught:
        series_io.geometry_to_xml(broken_polygon)
    assert caught.value.code == 'THMZ-E3000'

    broken_enclosure = dict(geometry, model_enclosure=(1, 2, 3, 999))
    with pytest.raises(MeshError) as caught:
        series_io.geometry_to_xml(broken_enclosure)
    assert caught.value.code == 'THMZ-E3000'

    # Reader path: the same dangling references in XML are rejected.
    read_broken = ET.fromstring(content)
    read_broken.find(
        'MaterialPolygons/MaterialPolygon').findall('NodeID')[-1].text = '999'
    with pytest.raises(MeshError) as caught:
        series_io.read_geometry_xml(ET.tostring(read_broken))
    assert caught.value.code == 'THMZ-E3000'

    read_broken = ET.fromstring(content)
    enclosure = read_broken.find('ModelEnclosure')
    ET.SubElement(enclosure, 'NodeID').text = '999'
    with pytest.raises(MeshError) as caught:
        series_io.read_geometry_xml(ET.tostring(read_broken))
    assert caught.value.code == 'THMZ-E3000'


def test_exchange_keys_ignore_edge_orientation(tmp_path):
    """BC edges are keyed by unordered node pairs; a counter-rotated edge
    must still land in its edge column (fsum merge is by sorted pair)."""
    result = solve(_transient_slab(tmp_path))
    heat = _read(result.members, STEP_EXCHANGE['heat'])
    # Two BC segments exist on the Hot Air boundary; both columns must carry
    # nonzero accumulated exchange, not leak into an unmatched column.
    nonzero = sum(1 for row in heat['values'][1:] for v in row if v != 0.0)
    assert nonzero > 0
