"""Ticket 03: fixed-temperature, flux and radiation boundary segments.

Each newly enabled BCSegments kind is verified against a 1-D analytic or
energy-balance benchmark on the single-material slab; out-of-scope segment
forms keep raising explicit coded rejections.
"""
from pathlib import Path
import zipfile
import xml.etree.ElementTree as ET

import pytest
from fairyfly_next_therm.result import THMZResult

from pyhygrothermfem_fairyfly import ContractError, UnsupportedError, run_thmz, solve
from pyhygrothermfem_fairyfly.run import STEFAN_BOLTZMANN

SLAB = Path(__file__).parent / 'assets' / 'slab.thmz'
ABS = 1e-6
# Slab: k=1, 2 m x 1 m; left boundary ID=2 'Hot Air', right ID=3
# 'Cold Air', bottom ID=4 and top ID=5 are 'Adiabatic'.
LEFT, RIGHT = '2', '3'
LEFT_EDGES = ((0, 3), (3, 6))
RIGHT_EDGES = ((2, 5), (5, 8))


def _load():
    with zipfile.ZipFile(SLAB) as archive:
        return {name: archive.read(name) for name in archive.namelist()}


def _write(members, path):
    with zipfile.ZipFile(path, 'w') as archive:
        for name, content in members.items():
            archive.writestr(name, content)
    return path


def _make_inert(members, name):
    """Reduce a BoundaryCondition record to an inert Simplified form."""
    conditions = ET.fromstring(members['SteadyStateBC.xml'])
    for record in conditions.findall('BoundaryCondition'):
        if record.findtext('Name') != name:
            continue
        for form in ('Comprehensive', 'Simplified'):
            element = record.find(form)
            if element is not None:
                record.remove(element)
        simple = ET.SubElement(record, 'Simplified')
        ET.SubElement(simple, 'Temperature').text = '0'
        ET.SubElement(simple, 'FilmCoefficient').text = '0'
        ET.SubElement(simple, 'RelativeHumidity').text = '0.5'
    members['SteadyStateBC.xml'] = ET.tostring(conditions)


def _drop_convections(members, labels):
    mesh = ET.fromstring(members['Mesh.xml'])
    convections = mesh.find('Case/BCSegments/Convections')
    for segment in list(convections):
        if segment.findtext('label') in labels:
            convections.remove(segment)
    members['Mesh.xml'] = ET.tostring(mesh)
    return mesh


def _segment_group(mesh, group):
    segments = mesh.find('Case/BCSegments')
    element = segments.find(group)
    if element is None:
        element = ET.SubElement(segments, group)
    return element


def _remove_convections_group(members):
    """Remove the whole Convections group, as a pure-Dirichlet export would."""
    mesh = ET.fromstring(members['Mesh.xml'])
    convections = mesh.find('Case/BCSegments/Convections')
    mesh.find('Case/BCSegments').remove(convections)
    members['Mesh.xml'] = ET.tostring(mesh)
    return mesh


def _add_temperatures(mesh, entries):
    """entries: (node, temperature, label) per mesh node."""
    group = _segment_group(mesh, 'Temperatures')
    for node, temperature, label in entries:
        segment = ET.SubElement(group, 'Temperature')
        ET.SubElement(segment, 'node').text = str(node)
        ET.SubElement(segment, 'temperature').text = str(temperature)
        ET.SubElement(segment, 'label').text = str(label)


def _add_fluxes(mesh, entries):
    """entries: (node1, node2, flux1, flux2, label)."""
    group = _segment_group(mesh, 'Fluxes')
    for index, (node1, node2, flux1, flux2, label) in enumerate(entries, 100):
        segment = ET.SubElement(group, 'Flux')
        ET.SubElement(segment, 'index').text = str(index)
        ET.SubElement(segment, 'node1').text = str(node1)
        ET.SubElement(segment, 'node2').text = str(node2)
        ET.SubElement(segment, 'flux1').text = str(flux1)
        ET.SubElement(segment, 'flux2').text = str(flux2)
        ET.SubElement(segment, 'label').text = str(label)


def _add_radiations(mesh, entries):
    """entries: (node1, node2, filmCoefficient, temperature)."""
    group = _segment_group(mesh, 'Radiations')
    for index, (node1, node2, coefficient, temperature) in enumerate(
            entries, 100):
        segment = ET.SubElement(group, 'Radiation')
        ET.SubElement(segment, 'index').text = str(index)
        ET.SubElement(segment, 'node1').text = str(node1)
        ET.SubElement(segment, 'node2').text = str(node2)
        ET.SubElement(segment, 'filmCoefficient').text = str(coefficient)
        ET.SubElement(segment, 'temperature1').text = str(temperature)
        ET.SubElement(segment, 'temperature2').text = str(temperature)


def _tag_boundaries(members, tags):
    model = ET.fromstring(members['Model.xml'])
    for boundary in model.findall('Boundaries/Boundary'):
        name = boundary.findtext('Name')
        if name in tags:
            boundary.find('FluxTag').text = tags[name]
    members['Model.xml'] = ET.tostring(model)


def _dirichlet_slab(path, tag=True):
    """Left/right edges fixed at 20/0 C; top/bottom stay adiabatic."""
    members = _load()
    _make_inert(members, 'Hot Air')
    _make_inert(members, 'Cold Air')
    mesh = _drop_convections(members, {LEFT, RIGHT})
    _add_temperatures(
        mesh,
        [(node, 20, LEFT) for node in (0, 3, 6)] +
        [(node, 0, RIGHT) for node in (2, 5, 8)])
    members['Mesh.xml'] = ET.tostring(mesh)
    if tag:
        _tag_boundaries(members, {'Hot Air': 'Hot', 'Cold Air': 'Cold'})
    return _write(members, path)


def _dirichlet_slab_no_convections(path, opaque=False, flux_zero=True,
                                   tag=True):
    """Dirichlet slab with the Convections group removed entirely.

    ``opaque=True`` relabels every BC segment with THERM-8-style opaque
    labels (e.g. 42/43) that match no Boundary/ID. Adiabatic top/bottom
    edges are declared as zero flux segments when ``flux_zero``.
    """
    members = _load()
    _make_inert(members, 'Hot Air')
    _make_inert(members, 'Cold Air')
    mesh = _remove_convections_group(members)
    hot, cold = ('42', '43') if opaque else (LEFT, RIGHT)
    _add_temperatures(
        mesh,
        [(node, 20, hot) for node in (0, 3, 6)] +
        [(node, 0, cold) for node in (2, 5, 8)])
    if flux_zero:
        _make_inert(members, 'Adiabatic')
        for u, v in ((0, 1), (1, 2), (6, 7), (7, 8)):
            _add_fluxes(mesh, [(u, v, 0.0, 0.0, '50' if opaque else
                                 ('4' if v <= 2 else '5'))])
    members['Mesh.xml'] = ET.tostring(mesh)
    if tag:
        _tag_boundaries(members, {'Hot Air': 'Hot', 'Cold Air': 'Cold'})
    return _write(members, path)


def _flux_slab(path, flux=-10.0, record_flux=None, tag=True):
    """Left edge convects at h=10/T=20; right edge carries uniform flux."""
    members = _load()
    _make_inert(members, 'Cold Air')
    if record_flux is not None:
        conditions = ET.fromstring(members['SteadyStateBC.xml'])
        record = next(r for r in conditions.findall('BoundaryCondition')
                      if r.findtext('Name') == 'Cold Air')
        record.remove(record.find('Simplified'))
        comprehensive = ET.SubElement(record, 'Comprehensive')
        ET.SubElement(comprehensive, 'RelativeHumidity').text = '0.5'
        convection = ET.SubElement(comprehensive, 'Convection')
        ET.SubElement(convection, 'Temperature').text = '0'
        ET.SubElement(convection, 'FilmCoefficient').text = '0'
        constant = ET.SubElement(comprehensive, 'ConstantFlux')
        ET.SubElement(constant, 'Flux').text = str(record_flux)
        members['SteadyStateBC.xml'] = ET.tostring(conditions)
    mesh = _drop_convections(members, {RIGHT})
    _add_fluxes(mesh, [(u, v, flux, flux, RIGHT) for u, v in RIGHT_EDGES])
    members['Mesh.xml'] = ET.tostring(mesh)
    if tag:
        _tag_boundaries(members, {'Hot Air': 'Hot', 'Cold Air': 'Cold'})
    return _write(members, path)


def _radiation_slab(path, emissivity=0.84, radiant_temperature=0.0, tag=True):
    """Left edge convects; right edge is a black-body radiation segment."""
    members = _load()
    _make_inert(members, 'Cold Air')
    mesh = _drop_convections(members, {RIGHT})
    _add_radiations(mesh, [
        (u, v, emissivity * STEFAN_BOLTZMANN, radiant_temperature)
        for u, v in RIGHT_EDGES])
    members['Mesh.xml'] = ET.tostring(mesh)
    if tag:
        _tag_boundaries(members, {'Hot Air': 'Hot', 'Cold Air': 'Cold'})
    return _write(members, path)


def test_fixed_temperature_edges_solve_dirichlet_slab(tmp_path):
    """Dirichlet slab: T = 20 - 10 x, q = 10 W/m2, energy closure by tag."""
    result = THMZResult(run_thmz(_dirichlet_slab(tmp_path / 'fixed.thmz')))
    assert result.temperatures == pytest.approx([20, 10, 0] * 3, abs=ABS)
    assert [q.x for q in result.heat_fluxes] == pytest.approx([10] * 9, abs=ABS)
    hot, cold = result.u_factors
    assert [u.name for u in result.u_factors] == ['Hot', 'Cold']
    # Element-side conduction integral: -10 W/m at x=0, +10 W/m at x=2.
    assert hot.heat_flux == pytest.approx(-10, abs=ABS)
    assert cold.heat_flux == pytest.approx(10, abs=ABS)
    assert hot.heat_flux + cold.heat_flux == pytest.approx(0, abs=ABS)
    assert hot.delta_temperature == 20
    assert hot.total_u_factor == pytest.approx(0.5, abs=ABS)


def test_fixed_temperature_result_available_from_solve_handle(tmp_path):
    source = _dirichlet_slab(tmp_path / 'fixed.thmz')
    result = solve(source)
    assert list(result.temperatures) == pytest.approx([20, 10, 0] * 3, abs=ABS)


def test_flux_segments_solve_energy_balance_slab(tmp_path):
    """Inward flux of -10 W/m2 on the cold edge leaves the slab at q=10."""
    result = THMZResult(run_thmz(
        _flux_slab(tmp_path / 'flux.thmz', flux=-10.0, record_flux=-10.0)))
    assert result.temperatures == pytest.approx([19, 9, -1] * 3, abs=ABS)
    assert [q.x for q in result.heat_fluxes] == pytest.approx([10] * 9, abs=ABS)
    hot, cold = result.u_factors
    # Outward signed heat: convection -10 W/m, flux edge +10 W/m.
    assert hot.heat_flux == pytest.approx(-10, abs=ABS)
    assert cold.heat_flux == pytest.approx(10, abs=ABS)
    # Only convection air temperatures feed the ambient pool: DeltaT is 0,
    # so no U-factor is published (apparent heat ratio is not a rating).
    assert hot.delta_temperature == 0
    assert hot.total_u_factor is None


def test_radiation_segments_close_the_energy_balance(tmp_path):
    """A black-body cold edge must emit exactly the conducted flux."""
    emissivity = 0.84
    result = THMZResult(run_thmz(
        _radiation_slab(tmp_path / 'radiation.thmz', emissivity)))
    surface = result.temperatures[2]  # Cold-edge nodes share one temperature.
    assert result.temperatures == pytest.approx(
        [result.temperatures[i] for i in range(3)] * 3, abs=ABS)
    interior = result.temperatures[0]
    conducted = 10 * (20 - interior)  # W/m2 through the hot film
    emitted = emissivity * STEFAN_BOLTZMANN * (
        (surface + 273.15) ** 4 - 273.15 ** 4)
    assert emitted == pytest.approx(conducted, rel=1e-6)
    assert interior < 20 and surface < interior
    hot, cold = result.u_factors
    assert hot.heat_flux + cold.heat_flux == pytest.approx(0, abs=1e-4)
    assert hot.delta_temperature == 20
    assert cold.delta_temperature == 20


@pytest.mark.parametrize('fault,code', [
    ('nonuniform_temperature', 'THMZ-E5015'),
    ('lone_temperature_node', 'THMZ-E3001'),
    ('temperature_label_mismatch', 'THMZ-E3001'),
    ('temperature_unknown_label', 'THMZ-E4001'),
    ('nonuniform_flux', 'THMZ-E5015'),
    ('flux_record_disagrees', 'THMZ-E4005'),
    ('mixed_kinds', 'THMZ-E4005'),
    ('active_radiation_enclosure', 'THMZ-E5006'),
    ('nonuniform_radiant_temperature', 'THMZ-E5015'),
])
def test_unsupported_or_inconsistent_segments_keep_coded_rejections(
        tmp_path, fault, code):
    """Rejected segment forms still raise explicit contract errors."""
    members = _load()
    _make_inert(members, 'Hot Air')
    _make_inert(members, 'Cold Air')
    mesh = _drop_convections(members, {LEFT, RIGHT})
    if fault == 'nonuniform_temperature':
        _add_temperatures(
            mesh, [(0, 20, LEFT), (3, 20, LEFT), (6, 21, LEFT)] +
            [(node, 0, RIGHT) for node in (2, 5, 8)])
    elif fault == 'lone_temperature_node':
        _add_temperatures(mesh, [(0, 20, LEFT), (3, 20, LEFT)])
    elif fault == 'temperature_label_mismatch':
        _add_temperatures(mesh, [(0, 20, LEFT), (3, 20, RIGHT)])
    elif fault == 'temperature_unknown_label':
        _add_temperatures(mesh, [(0, 20, '99'), (3, 20, '99'), (6, 20, '99')])
    elif fault == 'nonuniform_flux':
        _add_fluxes(mesh, [(u, v, -10, -5, RIGHT) for u, v in RIGHT_EDGES])
    elif fault == 'flux_record_disagrees':
        _add_fluxes(mesh, [(u, v, -10, -10, RIGHT) for u, v in RIGHT_EDGES])
    elif fault == 'mixed_kinds':
        # Edge (0,3) gets an inert convection segment while (3,6) is fixed-T
        # on the same boundary: two segment kinds on one boundary.
        convections = mesh.find('Case/BCSegments/Convections')
        segment = ET.SubElement(convections, 'Convection')
        ET.SubElement(segment, 'index').text = '100'
        ET.SubElement(segment, 'node1').text = '0'
        ET.SubElement(segment, 'node2').text = '3'
        ET.SubElement(segment, 'filmCoefficient').text = '0'
        ET.SubElement(segment, 'temperature1').text = '0'
        ET.SubElement(segment, 'temperature2').text = '0'
        ET.SubElement(segment, 'label').text = LEFT
        _add_temperatures(mesh, [(3, 20, LEFT), (6, 20, LEFT)])
    elif fault == 'active_radiation_enclosure':
        enclosure = ET.SubElement(_segment_group(mesh, 'RadiationEnclosures'),
                                  'RadiationEnclosure')
        ET.SubElement(enclosure, 'index').text = '1'
        ET.SubElement(enclosure, 'node1').text = '2'
        ET.SubElement(enclosure, 'node2').text = '5'
        ET.SubElement(enclosure, 'condCode').text = '0'
        ET.SubElement(enclosure, 'emissivity').text = '0.84'
        ET.SubElement(enclosure, 'temperature').text = '0'
    elif fault == 'nonuniform_radiant_temperature':
        _add_radiations(mesh, [(2, 5, 4.763e-08, 0)])
        group = mesh.find('Case/BCSegments/Radiations')
        last = list(group)[-1]
        last.find('temperature2').text = '-5'
    members['Mesh.xml'] = ET.tostring(mesh)
    source = _write(members, tmp_path / 'invalid.thmz')
    if fault == 'flux_record_disagrees':
        conditions = ET.fromstring(members['SteadyStateBC.xml'])
        record = next(r for r in conditions.findall('BoundaryCondition')
                      if r.findtext('Name') == 'Cold Air')
        record.remove(record.find('Simplified'))
        comprehensive = ET.SubElement(record, 'Comprehensive')
        ET.SubElement(comprehensive, 'RelativeHumidity').text = '0.5'
        convection = ET.SubElement(comprehensive, 'Convection')
        ET.SubElement(convection, 'Temperature').text = '0'
        ET.SubElement(convection, 'FilmCoefficient').text = '0'
        constant = ET.SubElement(comprehensive, 'ConstantFlux')
        ET.SubElement(constant, 'Flux').text = '-5'
        members['SteadyStateBC.xml'] = ET.tostring(conditions)
        _write(members, source)
    with pytest.raises(ContractError) as caught:
        run_thmz(source, output=tmp_path / 'out.thmz')
    assert caught.value.code == code
    if code.startswith('THMZ-E5'):
        assert isinstance(caught.value, UnsupportedError)
    assert not (tmp_path / 'out.thmz').exists()


# --- real-THERM label semantics --------------------------------------------

def test_dirichlet_slab_solves_without_convection_group(tmp_path):
    """A THMZ whose BCSegments carry no Convections group must still solve.

    THERM 8.1 always emits convection segments (adiabatic edges as film=0),
    but a conforming archive may declare Dirichlet, flux and radiation
    segments only; the solver must not require a convection block.
    """
    result = THMZResult(run_thmz(
        _dirichlet_slab_no_convections(tmp_path / 'fixed.thmz')))
    assert result.temperatures == pytest.approx([20, 10, 0] * 3, abs=ABS)
    hot, cold = result.u_factors
    assert hot.heat_flux + cold.heat_flux == pytest.approx(0, abs=ABS)


def test_opaque_therm_labels_resolve_by_geometry(tmp_path):
    """Segments labelled with opaque THERM indices (no Boundary/ID match)
    resolve their boundary geometrically, label groups staying coherent."""
    result = THMZResult(run_thmz(_dirichlet_slab_no_convections(
        tmp_path / 'opaque.thmz', opaque=True)))
    assert result.temperatures == pytest.approx([20, 10, 0] * 3, abs=ABS)
    hot, cold = result.u_factors
    assert hot.heat_flux == pytest.approx(-10, abs=ABS)
    assert cold.heat_flux == pytest.approx(10, abs=ABS)


def test_opaque_convection_and_flux_labels(tmp_path):
    """Opaque labels on Convection and Flux segments fall back to geometry."""
    members = _load()
    _make_inert(members, 'Cold Air')
    mesh = _drop_convections(members, {RIGHT})
    _add_fluxes(mesh, [(u, v, -10.0, -10.0, '99') for u, v in RIGHT_EDGES])
    for segment in mesh.find('Case/BCSegments/Convections'):
        segment.find('label').text = {
            LEFT: '31', RIGHT: '99', '4': '44', '5': '45'}[
            segment.findtext('label')]
    members['Mesh.xml'] = ET.tostring(mesh)
    conditions = ET.fromstring(members['SteadyStateBC.xml'])
    record = next(r for r in conditions.findall('BoundaryCondition')
                  if r.findtext('Name') == 'Cold Air')
    record.remove(record.find('Simplified'))
    comprehensive = ET.SubElement(record, 'Comprehensive')
    ET.SubElement(comprehensive, 'RelativeHumidity').text = '0.5'
    convection = ET.SubElement(comprehensive, 'Convection')
    ET.SubElement(convection, 'Temperature').text = '0'
    ET.SubElement(convection, 'FilmCoefficient').text = '0'
    constant = ET.SubElement(comprehensive, 'ConstantFlux')
    ET.SubElement(constant, 'Flux').text = '-10'
    members['SteadyStateBC.xml'] = ET.tostring(conditions)
    source = _write(members, tmp_path / 'opaque-flux.thmz')
    result = THMZResult(run_thmz(source))
    assert result.temperatures == pytest.approx([19, 9, -1] * 3, abs=ABS)


def test_opaque_label_geometry_contradiction_is_coded(tmp_path):
    """An opaque label whose nodes span two boundaries stays an error."""
    members = _load()
    _make_inert(members, 'Hot Air')
    _make_inert(members, 'Cold Air')
    mesh = _remove_convections_group(members)
    _add_temperatures(
        mesh,
        [(0, 20, '42'), (3, 20, '42'), (6, 20, '42'), (2, 20, '42')] +
        [(node, 0, '43') for node in (5, 8)])
    _add_fluxes(mesh, [
        (u, v, 0.0, 0.0, '50') for u, v in ((0, 1), (1, 2), (6, 7), (7, 8))])
    members['Mesh.xml'] = ET.tostring(mesh)
    source = _write(members, tmp_path / 'mixed.thmz')
    with pytest.raises(ContractError) as caught:
        run_thmz(source, output=tmp_path / 'out.thmz')
    assert caught.value.code == 'THMZ-E4003'
    assert not (tmp_path / 'out.thmz').exists()
