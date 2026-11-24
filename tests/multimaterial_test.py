"""Independent manufactured reference for the three-material T junction.

The reference is the exact source-free field T = 10 + 5 X Y / k, with
q = -k grad(T). Convection data are sampled per native exterior Q4 edge, so
the only approximation is the piecewise-constant Robin ambient on that edge.
"""
from collections import defaultdict
import argparse
import hashlib
import json
import math
from pathlib import Path
import tempfile
import uuid
import xml.etree.ElementTree as ET
import zipfile

import pytest

from pyhygrothermfem_fairyfly import run_thmz


REPO = Path(__file__).resolve().parents[1]
ASSETS = Path(__file__).resolve().parent / 'assets'
NATIVE_TARGETS = (80, 320, 1280, 5120)
REGION_K = {'left': 1.0, 'lower_right': 2.0, 'upper_right': 4.0}
FILM = 10.0
T0 = 10.0
AMPLITUDE = 5.0
HALF_SIDE_REFERENCE = {
    ('left', 'lower'): -2.5, ('left', 'upper'): 2.5,
    ('right', 'lower'): 2.5, ('right', 'upper'): -2.5,
    ('bottom', 'left'): -2.5, ('bottom', 'right'): 2.5,
    ('top', 'left'): 2.5, ('top', 'right'): -2.5,
}
_UUID_NAMESPACE = uuid.UUID('8c636a6d-423e-4a16-97af-c61a6ef04be2')


def _number(parent, path, field):
    try:
        value = float(parent.findtext(path))
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError('{} must be numeric'.format(field)) from error
    if not math.isfinite(value):
        raise ValueError('{} must be finite'.format(field))
    return value


def _region(x, y):
    """Return region for a point away from an interface."""
    if x < 0:
        return 'left'
    return 'lower_right' if y < 0 else 'upper_right'


def _temperature(x, y, conductivity, half_x, half_y):
    return T0 + AMPLITUDE * (x / half_x) * (y / half_y) / conductivity


def _flux(x, y, half_x, half_y):
    return (-AMPLITUDE * (y / half_y) / half_x,
            -AMPLITUDE * (x / half_x) / half_y)


def _normal_and_group(points, bounds):
    (x1, y1), (x2, y2) = points
    min_x, max_x, min_y, max_y = bounds
    mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
    tolerance = 1e-6
    if abs(x1 - x2) <= tolerance:
        if abs(mid_x - min_x) <= tolerance:
            side, normal = 'left', (-1.0, 0.0)
        elif abs(mid_x - max_x) <= tolerance:
            side, normal = 'right', (1.0, 0.0)
        else:
            raise ValueError('Convection edge is not on a vertical exterior side')
        half = 'lower' if mid_y < 0 else 'upper'
    elif abs(y1 - y2) <= tolerance:
        if abs(mid_y - min_y) <= tolerance:
            side, normal = 'bottom', (0.0, -1.0)
        elif abs(mid_y - max_y) <= tolerance:
            side, normal = 'top', (0.0, 1.0)
        else:
            raise ValueError('Convection edge is not on a horizontal exterior side')
        half = 'left' if mid_x < 0 else 'right'
    else:
        raise ValueError('Native T mesh has a non-axis-aligned exterior edge')
    if ((side in ('left', 'right') and min(y1, y2) < -tolerance
         and max(y1, y2) > tolerance) or
            (side in ('top', 'bottom') and min(x1, x2) < -tolerance
             and max(x1, x2) > tolerance)):
        raise ValueError('Native boundary edge crosses the half-side split')
    return normal, (side, half)


def _text(element, name, value):
    child = element.find(name)
    if child is None:
        child = ET.SubElement(element, name)
    child.text = str(value)
    return child


def _boundary_condition(name, ambient, identifier):
    condition = ET.Element('BoundaryCondition')
    ET.SubElement(condition, 'UUID').text = identifier
    ET.SubElement(condition, 'Name').text = name
    ET.SubElement(condition, 'ProjectNameTag').text = ''
    ET.SubElement(condition, 'Protected').text = 'false'
    ET.SubElement(condition, 'Color').text = '0x000000'
    ET.SubElement(condition, 'IGUSurface').text = 'false'
    simple = ET.SubElement(condition, 'Simplified')
    ET.SubElement(simple, 'Temperature').text = format(ambient, '.15g')
    ET.SubElement(simple, 'FilmCoefficient').text = format(FILM, '.15g')
    ET.SubElement(simple, 'RelativeHumidity').text = '0.5'
    return condition


def make_manufactured_thmz(source, target):
    """Recenter a Rhino T mesh and apply exact Robin data per exterior Q4 edge.

    Geometry and Q4 topology are retained. The only mesh changes are a rigid
    coordinate translation and the convection fields/labels on existing edges.
    """
    source, target = Path(source).resolve(), Path(target).resolve()
    if source == target:
        raise ValueError('Manufactured input must be written to a separate THMZ')
    with zipfile.ZipFile(source) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    required = ('Model.xml', 'Mesh.xml', 'Materials.xml', 'SteadyStateBC.xml')
    if any(name not in members for name in required):
        raise ValueError('Native THMZ is missing a required model member')
    roots = {name: ET.fromstring(members[name]) for name in required}
    model, mesh, materials, conditions = (roots[name] for name in required)
    case = mesh.find('Case')
    nodes_xml = case.findall('Nodes/Node')
    elements_xml = case.findall('Elements/Element')
    bcs_xml = case.findall('BCSegments/Convections/Convection')
    if not nodes_xml or not elements_xml or not bcs_xml:
        raise ValueError('Native THMZ must contain Q4s and exterior convection edges')

    node_points = [(_number(node, 'x', 'Node x'), _number(node, 'y', 'Node y'))
                   for node in nodes_xml]
    min_x = min(point[0] for point in node_points)
    max_x = max(point[0] for point in node_points)
    min_y = min(point[1] for point in node_points)
    max_y = max(point[1] for point in node_points)
    center_x, center_y = (min_x + max_x) / 2, (min_y + max_y) / 2
    half_x, half_y = (max_x - min_x) / 2, (max_y - min_y) / 2
    if not math.isclose(half_x, 1.0, abs_tol=1e-6) or not math.isclose(
            half_y, 1.0, abs_tol=1e-6):
        raise ValueError('Manufactured T reference requires a 2 m by 2 m mesh')
    centered = [(x - center_x, y - center_y) for x, y in node_points]

    origin = model.find('GlazingOrigin')
    origin_x = _number(origin, 'x', 'GlazingOrigin x')
    origin_y = _number(origin, 'y', 'GlazingOrigin y')
    polygons = model.findall('Polygons/Polygon')
    if len(polygons) != 3:
        raise ValueError('T reference requires three material polygons')
    uuid_by_region, id_by_region, id_by_material, conductivity_by_uuid = {}, {}, {}, {}
    for polygon in polygons:
        material_uuid = (polygon.findtext('MaterialUUID') or '').casefold()
        polygon_uuid = (polygon.findtext('UUID') or '').casefold()
        points = [(_number(point, 'x', 'Polygon x'),
                   _number(point, 'y', 'Polygon y'))
                  for point in polygon.findall('Points/Point')]
        if not material_uuid or not polygon_uuid or len(points) < 3:
            raise ValueError('Each T region needs polygon and material UUIDs')
        polygon_x = ((min(point[0] for point in points) +
                      max(point[0] for point in points)) / 2 - origin_x) / 1000
        polygon_y = ((min(point[1] for point in points) +
                      max(point[1] for point in points)) / 2 - origin_y) / 1000
        region = _region(polygon_x - center_x, polygon_y - center_y)
        if region in uuid_by_region:
            raise ValueError('T reference has multiple polygons in region {}'.format(
                region))
        if material_uuid in conductivity_by_uuid:
            raise ValueError('T regions must have distinct material UUIDs')
        uuid_by_region[region] = material_uuid
        conductivity_by_uuid[material_uuid] = REGION_K[region]
        if material_uuid not in id_by_material:
            id_by_material[material_uuid] = len(id_by_material) + 1
        id_by_region[region] = id_by_material[material_uuid]
    if set(uuid_by_region) != set(REGION_K):
        raise ValueError(
            'Native polygons do not form left/lower-right/upper-right T regions')

    # Verify the Rhino exporter already assigned each face to its polygon.
    for element in elements_xml:
        refs = [int(_number(element, 'node{}'.format(i), 'Q4 node'))
                for i in range(1, 5)]
        if any(index < 0 or index >= len(centered) for index in refs):
            raise ValueError('Q4 has an out-of-range node reference')
        x = math.fsum(centered[index][0] for index in refs) / 4
        y = math.fsum(centered[index][1] for index in refs) / 4
        region = _region(x, y)
        if int(element.findtext('materialID')) != id_by_region[region]:
            raise ValueError('Q4 materialID does not match its T region')

    material_records = {}
    for material in materials.findall('Material'):
        key = (material.findtext('UUID') or '').casefold()
        if key:
            material_records.setdefault(key, []).append(material)
    for key, conductivity in conductivity_by_uuid.items():
        records = material_records.get(key, [])
        if len(records) != 1:
            raise ValueError('Material UUID {} must resolve once'.format(key))
        dry = records[0].find('Solid/HygroThermal/ThermalConductivityDry')
        if dry is None:
            raise ValueError('Material {} has no dry thermal conductivity'.format(key))
        dry.text = format(conductivity, '.15g')

    # Keep the local coordinate origin fixed while translating geometry and mesh.
    for node, (x, y) in zip(nodes_xml, centered):
        _text(node, 'x', format(x, '.15g'))
        _text(node, 'y', format(y, '.15g'))
    for polygon in polygons:
        for point in polygon.findall('Points/Point'):
            _text(point, 'x', format(_number(point, 'x', 'Polygon x') -
                                     center_x * 1000, '.15g'))
            _text(point, 'y', format(_number(point, 'y', 'Polygon y') -
                                     center_y * 1000, '.15g'))

    old_boundaries = {boundary.findtext('ID'): boundary
                      for boundary in model.findall('Boundaries/Boundary')}
    boundaries_container = model.find('Boundaries')
    if boundaries_container is None:
        raise ValueError('Model.xml is missing Boundaries')
    for child in list(boundaries_container):
        boundaries_container.remove(child)
    first_id = max((int(key) for key in old_boundaries), default=0) + 1
    condition_records = [child for child in list(conditions)
                         if child.tag != 'BoundaryCondition']
    for child in list(conditions):
        conditions.remove(child)
    for child in condition_records:
        conditions.append(child)

    bounds = (min(x for x, _ in centered), max(x for x, _ in centered),
              min(y for _, y in centered), max(y for _, y in centered))
    for index, bc in enumerate(bcs_xml, 1):
        old_label = bc.findtext('label')
        original = old_boundaries.get(old_label)
        if original is None:
            raise ValueError('Convection label {} has no Model boundary'.format(
                old_label))
        node1, node2 = int(bc.findtext('node1')), int(bc.findtext('node2'))
        try:
            p1, p2 = centered[node1], centered[node2]
        except IndexError as error:
            raise ValueError('Convection has an out-of-range node reference') from error
        normal, _group = _normal_and_group((p1, p2), bounds)
        mid_x, mid_y = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
        conductivity = REGION_K[_region(mid_x / half_x, mid_y / half_y)]
        air = []
        for x, y in (p1, p2):
            surface = _temperature(x, y, conductivity, half_x, half_y)
            qx, qy = _flux(x, y, half_x, half_y)
            air.append(surface - (qx * normal[0] + qy * normal[1]) / FILM)
        ambient = math.fsum(air) / 2
        new_id = str(first_id + index - 1)
        name = 'MREF-EDGE-{:04d}'.format(index)
        boundary = ET.fromstring(ET.tostring(original))
        _text(boundary, 'ID', new_id)
        _text(boundary, 'UUID', str(uuid.uuid5(
            _UUID_NAMESPACE, '{}:{}:{}'.format(source.name, node1, node2))))
        _text(boundary, 'Name', name)
        _text(boundary, 'FluxTag', 'Net balance')
        start, end = boundary.find('StartPoint'), boundary.find('EndPoint')
        for point, (x, y) in ((start, p1), (end, p2)):
            _text(point, 'x', format(x * 1000 + origin_x, '.15g'))
            _text(point, 'y', format(y * 1000 + origin_y, '.15g'))
        emissivity = boundary.find('ThermalEmissionProperties/Emissivity')
        if emissivity is not None:
            emissivity.text = '0'
        boundaries_container.append(boundary)

        _text(bc, 'filmCoefficient', format(FILM, '.15g'))
        _text(bc, 'temperature1', format(ambient, '.15g'))
        _text(bc, 'temperature2', format(ambient, '.15g'))
        _text(bc, 'label', new_id)
        conditions.append(_boundary_condition(
            name, ambient, str(uuid.uuid5(_UUID_NAMESPACE,
                                         '{}:bc:{}'.format(source.name, index)))))

    replacements = {
        'Model.xml': model, 'Mesh.xml': mesh,
        'Materials.xml': materials, 'SteadyStateBC.xml': conditions,
    }
    for name, root in replacements.items():
        members[name] = ET.tostring(root, encoding='utf-8', xml_declaration=True)
    for name in ('SteadyStateResults.xml', 'SteadyStateMeshResults.xml',
                 'pyHygroTherm.json'):
        members.pop(name, None)
    for name in tuple(members):
        if name.startswith('transient results/'):
            members.pop(name)
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    return target


def analyze_manufactured_result(path):
    """Return temperature error, Robin integrals and the writer's net balance."""
    with zipfile.ZipFile(path) as archive:
        mesh = ET.fromstring(archive.read('Mesh.xml'))
        results = ET.fromstring(archive.read('SteadyStateMeshResults.xml'))
        steady = ET.fromstring(archive.read('SteadyStateResults.xml'))
    case = mesh.find('Case')
    nodes_xml = case.findall('Nodes/Node')
    result_nodes = results.findall('Case/Nodes/Node')
    if len(nodes_xml) != len(result_nodes):
        raise ValueError('Result node count does not match the native mesh')
    coordinates = [(float(node.findtext('x')), float(node.findtext('y')))
                   for node in nodes_xml]
    result_by_label = {int(node.findtext('Index')): float(node.findtext('Temperature'))
                       for node in result_nodes}
    temperatures = [result_by_label[int(node.findtext('index'))]
                    for node in nodes_xml]
    min_x, max_x = min(x for x, _ in coordinates), max(x for x, _ in coordinates)
    min_y, max_y = min(y for _, y in coordinates), max(y for _, y in coordinates)
    half_x, half_y = (max_x - min_x) / 2, (max_y - min_y) / 2
    expected = [
        _temperature(x, y, REGION_K[_region(x / half_x, y / half_y)],
                     half_x, half_y)
        for x, y in coordinates]
    max_temperature_error = max(abs(actual - ref)
                                for actual, ref in zip(temperatures, expected))

    bounds = (min_x, max_x, min_y, max_y)
    integrals = defaultdict(list)
    for bc in case.findall('BCSegments/Convections/Convection'):
        first, second = int(bc.findtext('node1')), int(bc.findtext('node2'))
        p1, p2 = coordinates[first], coordinates[second]
        normal, group = _normal_and_group((p1, p2), bounds)
        h = float(bc.findtext('filmCoefficient'))
        ambient = (float(bc.findtext('temperature1')) +
                   float(bc.findtext('temperature2'))) / 2
        length = math.dist(p1, p2)
        mean_surface = (temperatures[first] + temperatures[second]) / 2
        integrals[group].append(h * length * (mean_surface - ambient))
    signed_integrals = {key: math.fsum(values) for key, values in integrals.items()}
    if set(signed_integrals) != set(HALF_SIDE_REFERENCE):
        raise ValueError('Manufactured mesh does not cover all eight half-sides')
    errors = {key: abs(signed_integrals[key] - expected_value) /
              abs(expected_value)
              for key, expected_value in HALF_SIDE_REFERENCE.items()}
    net_records = [item for item in steady.findall('Case/U-factors')
                   if item.findtext('Tag') == 'Net balance']
    if len(net_records) != 1:
        raise ValueError('Result must contain one Net balance integration')
    return {
        'nodes': len(nodes_xml),
        'elements': len(case.findall('Elements/Element')),
        'max_temperature_error_C': max_temperature_error,
        'half_side_outward_heat_W_per_m': {
            '{}_{}'.format(*key): value for key, value in signed_integrals.items()},
        'max_half_side_relative_error': max(errors.values()),
        'net_outward_heat_W_per_m': float(net_records[0].findtext('HeatFlux')),
        'rounded_mesh_result_net_W_per_m': math.fsum(signed_integrals.values()),
        'half_x_m': half_x,
        'half_y_m': half_y,
    }


def _run_cases(sources, output_dir):
    results = {}
    for source in sources:
        target = Path(output_dir) / (source.stem + '_manufactured.thmz')
        solved = target.with_name(target.stem + '_solved.thmz')
        make_manufactured_thmz(source, target)
        run_thmz(target, output=solved)
        metrics = analyze_manufactured_result(solved)
        metrics['temperature_reference_pass'] = (
            metrics['max_temperature_error_C'] <= 0.1)
        metrics['half_side_reference_pass'] = (
            metrics['max_half_side_relative_error'] <= 0.005)
        metrics['source_sha256'] = hashlib.sha256(source.read_bytes()).hexdigest()
        metrics['source_path'] = str(source.relative_to(REPO))
        metrics['manufactured_input_sha256'] = hashlib.sha256(
            target.read_bytes()).hexdigest()
        metrics['solved_output_sha256'] = hashlib.sha256(
            solved.read_bytes()).hexdigest()
        results[source.stem] = metrics
    return results


def _native_sources():
    paths = [ASSETS / 'ticket07_junction_{}.thmz'.format(target)
             for target in NATIVE_TARGETS]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError('Missing fixed Rhino-native fixtures: {}'.format(
            ', '.join(str(path) for path in missing)))
    return paths


def _assert_reference(results):
    for case, metrics in results.items():
        assert metrics['net_outward_heat_W_per_m'] == pytest.approx(
            0.0, abs=1e-6), '{} boundary heat does not balance'.format(case)
    assert any(case.endswith('_1280') for case in results), \
        '1280 target is retained to report its failed refinement result'
    fine = next((metrics for case, metrics in results.items()
                 if case.endswith('_5120')), None)
    assert fine is not None, 'Fine (5120 target) native T mesh is required'
    assert fine['max_temperature_error_C'] <= 0.1
    assert fine['max_half_side_relative_error'] <= 0.005


def test_native_t_junction_matches_independent_manufactured_reference(tmp_path):
    """Run all four fixed Rhino-native meshes through public THMZ execution."""
    sources = _native_sources()
    results = _run_cases(sources, tmp_path)
    _assert_reference(results)
    print(json.dumps(results, indent=2, sort_keys=True))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path,
                        help='keep manufactured inputs/results and report here')
    args = parser.parse_args(argv)
    sources = _native_sources()
    if args.output_dir:
        output_dir = args.output_dir.resolve()
        results = _run_cases(sources, output_dir)
        _assert_reference(results)
        report = {
            'reference': {
                'case': 'three-material T junction manufactured exact solution',
                'conductivity_W_per_mK': REGION_K,
                'temperature_C': '10 + 5*X*Y/k',
                'heat_flux_W_per_m2': '(-5*Y/Lx, -5*X/Ly)',
                'boundary_approximation': (
                    'Per-edge uniform h=10 W/(m2.K); air temperature is the '
                    'mean exact Robin value at the native Q4 edge endpoints.'),
                'fine_limits': {
                    'temperature_max_abs_C': 0.1,
                    'half_side_heat_max_relative': 0.005,
                    'net_heat_abs_W_per_m': 1e-6,
                },
            },
            'cases': results,
        }
        (output_dir / 'manufactured-validation.json').write_text(
            json.dumps(report, indent=2, sort_keys=True) + '\n',
            encoding='utf-8')
        print(json.dumps(results, indent=2, sort_keys=True))
    else:
        with tempfile.TemporaryDirectory(prefix='t-junction-reference-') as directory:
            results = _run_cases(sources, directory)
            _assert_reference(results)
            print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
