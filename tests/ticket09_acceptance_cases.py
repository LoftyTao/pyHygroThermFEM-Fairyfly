"""Premeshed, independently decoded THMZ cases for ticket-09 acceptance.

The fixed slab archive supplies the documented container skeleton only.
No solver parser, native mesher, or production input helper is called.
"""
import csv
from hashlib import sha256
import io
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile
from ticket18_boundary_fixtures import declare_constant_boundaries


ASSET = Path(__file__).parent / 'assets' / 'slab.thmz'
MATERIAL_UUID = '8f20094a-3c24-4258-a319-30b7029039a0'
PIN_UUID = 'ticket09-temperature-humidity-pin'
INVALID_MATERIAL_VALUES = (
    ('BulkDensity', -600.25),
    ('Porosity', 1.125),
    ('SpecificHeatCapacityDry', -850.5),
    ('WaterVaporDiffusionResistanceFactor', -50.75),
)


def file_digest(path):
    """Return the SHA-256 of the exact file bytes."""
    return sha256(Path(path).read_bytes()).hexdigest()


def _put(parent, tag, value):
    child = parent.find(tag)
    if child is None:
        child = ET.SubElement(parent, tag)
    child.text = str(value)
    return child


def wet_slab(path, *, nx=1, ny=1, temperature_pin=(30.0, 5.0),
             initial_temperature=10.0, initial_humidity=0.5,
             time_step=1.0, num_steps=2, invalid_material=None):
    """Write a 2x1 m wet slab with an explicit regular Q4 mesh.

    Args:
        path: Destination THMZ path.
        nx: Number of rectangular elements in the x direction.
        ny: Number of rectangular elements in the y direction.
        temperature_pin: Positive-time pin series; first entry seeds t=0.
        initial_temperature: Declared uniform initial temperature in C.
        initial_humidity: Declared initial relative humidity fraction.
        time_step: Real positive timestep duration in seconds.
        num_steps: Number of positive-time steps to compute.
        invalid_material: Optional (XML field, value) rejection mutation.

    Returns:
        pathlib.Path: Written input archive.
    """
    with zipfile.ZipFile(ASSET) as archive:
        members = {name: archive.read(name) for name in archive.namelist()
                   if not name.startswith('transient results/')
                   and name not in ('SteadyStateResults.xml',
                                    'SteadyStateMeshResults.xml',
                                    'pyHygroTherm.json')}
    model = ET.fromstring(members['Model.xml'])
    options = model.find('Properties/CalculationOptions')
    _put(options, 'CalculationMode', 'Transient')
    _put(options, 'SimulateThermal', 'true')
    _put(options, 'SimulateMoisture', 'true')
    _put(options, 'TransientCalculationMethodology', 'Constant everywhere')
    initial = options.find('ConstantInitialConditionsTransient')
    _put(initial, 'Temperature', initial_temperature)
    _put(initial, 'Humidity', initial_humidity)
    engine = options.find('EngineParameters')
    for tag, value in (('TimeStep', time_step),
                       ('NumberOfTimesteps', num_steps),
                       ('ConvergenceTolerance', '1e-10'),
                       ('RelaxationParameter', '1')):
        _put(engine, tag, value)
    modeling = options.find('ModelingOptions')
    if modeling is None:
        modeling = ET.SubElement(options, 'ModelingOptions')
    for tag in ('ExcludeWaterLiquidTransportation',
                'ExcludeHeatOfEvaporation', 'ExcludeCapillaryConduction',
                'ExcludeVaporDiffusionConduction', 'ExcludeLatentHeatOfFusion',
                'ExcludeThermalConductivityMoistureDependent',
                'ExcludeThermalConductivityTemperatureDependent'):
        _put(modeling, tag, 'true')
    hot = next(boundary for boundary in model.findall('Boundaries/Boundary')
               if boundary.findtext('ID') == '2')
    transient = ET.SubElement(hot, 'TransientData')
    _put(transient, 'UUID', PIN_UUID)
    _put(transient, 'TimestepFileName', 'ticket09-pin.xml')
    members['Model.xml'] = ET.tostring(model)

    materials = ET.fromstring(members['Materials.xml'])
    hygro = materials.find('Material/Solid/HygroThermal')
    for tag, value in (('BulkDensity', 600.0), ('Porosity', 0.2),
                       ('SpecificHeatCapacityDry', 850.0),
                       ('WaterVaporDiffusionResistanceFactor', 50.0)):
        _put(hygro, tag, repr(value))
    storage = ET.SubElement(hygro, 'MoistureStorageFunction')
    for humidity, water in ((0.0, 0.0), (0.5, 75.0), (1.0, 150.0)):
        point = ET.SubElement(storage, 'TableValue')
        _put(point, 'x', humidity)
        _put(point, 'y', water)
    if invalid_material is not None:
        _put(hygro, invalid_material[0], repr(invalid_material[1]))
    members['Materials.xml'] = ET.tostring(materials)

    bcs = ET.fromstring(members['SteadyStateBC.xml'])
    for record in bcs.findall('BoundaryCondition'):
        for block in list(record):
            humidity = block.find('RelativeHumidity')
            if humidity is not None:
                block.remove(humidity)
    # Ticket 26: a Neumann convection edge always carries the moisture
    # Robin pair in moist solves, so a steady transient record needs its
    # Humidity scalar as phi_air. Re-declare it on Cold Air at the
    # initial field value; these tests assert only the t=0 frame.
    for record in bcs.findall('BoundaryCondition'):
        if record.findtext('Name') != 'Cold Air':
            continue
        for block in list(record):
            if block.tag in ('Simplified', 'Comprehensive'):
                ET.SubElement(block, 'RelativeHumidity').text = '0.5'
    members['SteadyStateBC.xml'] = ET.tostring(bcs)

    types = ET.Element('BoundaryConditionsType')
    _put(types, 'Version', '1')
    record = ET.SubElement(types, 'BoundaryConditionType')
    for tag, value in (('UUID', PIN_UUID), ('Name', 'Ticket09 wet pin'),
                       ('Protected', 'false'), ('bcType', 'Transient'),
                       ('bcModel', 'Dirichlet'), ('UseHeatFlux', 'false'),
                       ('UseTemperature', 'true'), ('UseHumidity', 'true'),
                       ('Color', '0x08c4cd')):
        _put(record, tag, value)
    members['TransientTypeBC.xml'] = ET.tostring(types)

    timestep = ET.Element('InputBoundaryConditionsData')
    _put(timestep, 'Version', '1')
    _put(timestep, 'Name', 'Ticket09 wet pin')
    entries = ET.SubElement(timestep, 'BoundaryConditionTransient')
    for index, temperature in enumerate(temperature_pin, 1):
        point = ET.SubElement(entries, 'TemperatureTimestep')
        _put(point, 'Index', index)
        _put(point, 'Temperature', temperature)
    humidity = ET.SubElement(entries, 'HumidityTimestep')
    _put(humidity, 'Index', 1)
    _put(humidity, 'Humidity', initial_humidity)
    members['timestep input files/ticket09-pin.xml'] = ET.tostring(timestep)

    mesh = ET.Element('Mesh')
    _put(mesh, 'Version', '1')
    case = ET.SubElement(mesh, 'Case')
    _put(case, 'ModelType', 'U-factor')
    nodes = ET.SubElement(case, 'Nodes')
    for row in range(ny + 1):
        for col in range(nx + 1):
            node = ET.SubElement(nodes, 'Node')
            _put(node, 'index', row * (nx + 1) + col + 1)
            _put(node, 'x', col * 2.0 / nx)
            _put(node, 'y', row * 1.0 / ny)
    elements = ET.SubElement(case, 'Elements')
    for row in range(ny):
        for col in range(nx):
            a = row * (nx + 1) + col
            element = ET.SubElement(elements, 'Element')
            _put(element, 'index', row * nx + col + 1)
            for tag, index in zip(('node1', 'node2', 'node3', 'node4'),
                                  (a, a + 1, a + nx + 2, a + nx + 1)):
                _put(element, tag, index)
            _put(element, 'materialID', 1)
    segments = ET.SubElement(case, 'BCSegments')
    convections = ET.SubElement(segments, 'Convections')
    boundaries = []
    for row in range(ny):
        low = row * (nx + 1)
        high = (row + 1) * (nx + 1)
        boundaries.extend(((high, low, 2, 10.0, 20.0),
                           (low + nx, high + nx, 3, 10.0, 0.0)))
    for col in range(nx):
        top = ny * (nx + 1) + col
        boundaries.extend(((col, col + 1, 4, 0.0, 0.0),
                           (top, top + 1, 5, 0.0, 0.0)))
    for index, (a, b, label, film, temperature) in enumerate(boundaries, 1):
        edge = ET.SubElement(convections, 'Convection')
        for tag, value in (('index', index), ('node1', a), ('node2', b),
                           ('filmCoefficient', film),
                           ('temperature1', temperature),
                           ('temperature2', temperature), ('label', label)):
            _put(edge, tag, value)
    members['Mesh.xml'] = ET.tostring(mesh)

    declare_constant_boundaries(members)

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, content in members.items():
            archive.writestr(name, content)
    return target


def csv_rows(content):
    """Decode numeric archive CSV independently of the solver reader."""
    reader = csv.reader(io.StringIO(content.decode('utf-8-sig')))
    header = next(reader)
    if header[0] != 'Time (s)':
        raise ValueError('Expected the declared pyHygroTherm time column')
    return header, tuple(tuple(float(value) for value in row)
                         for row in reader)


def t0_observation(members):
    """Return first-frame data without obtaining any later solved field."""
    values = {}
    for name in ('Temperatures', 'Humidities', 'HeatFlux', 'WaterFlux',
                 'HeatFluxEdges', 'WaterFluxEdges'):
        header, rows = csv_rows(members['transient results/' + name + '.csv'])
        if rows[0][0] != 0.0:
            raise ValueError('The first archive frame must be at t=0')
        values[name] = {'entities': header[1:], 'values': rows[0][1:]}
    return values


def _table(parent, tag, points):
    table = parent.find(tag)
    if table is None:
        table = ET.SubElement(parent, tag)
    table.clear()
    for x, y in points:
        point = ET.SubElement(table, 'TableValue')
        _put(point, 'x', repr(x))
        _put(point, 'y', repr(y))


def physics_slab(path, *, nx=32, ny=8, width=0.2, height=0.05,
                 storage=2.0, mu=1.0, density=600.0, heat_capacity=850.0,
                 initial_temperature=20.0, initial_humidity=0.5,
                 time_step=60.0, duration=21600.0, left=None, right=None,
                 liquid=None, conductivity_water=None,
                 conductivity_temperature=None, fusion=False,
                 evaporation=False, vapor_heat=False):
    """Write a documented rectangular benchmark without invoking a mesher.

    Args:
        path: Destination THMZ path.
        nx: Number of elements across the width.
        ny: Number of elements across the height.
        width: Rectangle width in m.
        height: Rectangle height in m.
        storage: Linear sorption slope in kg/m3 per RH fraction.
        mu: Dimensionless vapor diffusion resistance factor.
        density: Dry bulk density in kg/m3.
        heat_capacity: Dry specific heat in J/(kg.K).
        initial_temperature: Uniform initial temperature in C.
        initial_humidity: Uniform initial RH fraction.
        time_step: Positive implicit step duration in s.
        duration: Exact total duration in s, divisible by time_step.
        left: Boundary mapping: mode, temperature, humidity, film.
            mode='pin' fixes temperature and optionally humidity;
            mode='robin' exchanges heat and optionally vapor.
        right: Boundary mapping with the same fields as left.
        liquid: Constant liquid transport coefficient in m2/s, or None.
        conductivity_water: (w, k) table, kg/m3 and W/(m.K), or None.
        conductivity_temperature: (T, k) table, C and W/(m.K), or None.
        fusion: Whether to include the declared freezing model.
        evaporation: Whether to include evaporation heat.
        vapor_heat: Whether to include vapor sensible-heat transport.

    Returns:
        pathlib.Path: Written premeshed archive.
    """
    steps = round(duration / time_step)
    if steps <= 0 or abs(steps * time_step - duration) > 1e-9:
        raise ValueError('Duration must contain exact positive timesteps')
    if min(nx, ny, width, height, time_step) <= 0:
        raise ValueError('Mesh sizes, lengths and timestep must be positive')
    default = {'mode': 'robin', 'temperature': 20.0,
               'humidity': 0.9, 'film': 10.0}
    left = default if left is None else left
    right = default if right is None else right
    target = wet_slab(
        path, nx=nx, ny=ny, temperature_pin=(initial_temperature,),
        initial_temperature=initial_temperature,
        initial_humidity=initial_humidity, time_step=time_step,
        num_steps=steps)
    with zipfile.ZipFile(target) as archive:
        members = {name: archive.read(name) for name in archive.namelist()
                   if not name.startswith('timestep input files/')}
    model = ET.fromstring(members['Model.xml'])
    _put(model.find('GlazingOrigin'), 'x', 0)
    _put(model.find('GlazingOrigin'), 'y', 0)
    for point in model.findall('Polygons/Polygon/Points/Point'):
        _put(point, 'x', (float(point.findtext('x')) - 250) * width / 2)
        _put(point, 'y', (float(point.findtext('y')) + 250) * height)
    for boundary in model.findall('Boundaries/Boundary'):
        for tag in ('StartPoint', 'EndPoint'):
            point = boundary.find(tag)
            _put(point, 'x', (float(point.findtext('x')) - 250) * width / 2)
            _put(point, 'y', (float(point.findtext('y')) + 250) * height)
        data = boundary.find('TransientData')
        if data is not None:
            boundary.remove(data)
    modeling = model.find('Properties/CalculationOptions/ModelingOptions')
    _put(modeling, 'ExcludeWaterLiquidTransportation',
         str(liquid is None).lower())
    _put(modeling, 'ExcludeLatentHeatOfFusion', str(not fusion).lower())
    _put(modeling, 'ExcludeHeatOfEvaporation', str(not evaporation).lower())
    _put(modeling, 'ExcludeVaporDiffusionConduction',
         str(not vapor_heat).lower())
    variable = (conductivity_water is not None
                or conductivity_temperature is not None)
    for tag in ('ExcludeThermalConductivityMoistureDependent',
                'ExcludeThermalConductivityTemperatureDependent'):
        _put(modeling, tag, str(not variable).lower())
    materials = ET.fromstring(members['Materials.xml'])
    hygro = materials.find('Material/Solid/HygroThermal')
    for tag, value in (('BulkDensity', density),
                       ('SpecificHeatCapacityDry', heat_capacity),
                       ('WaterVaporDiffusionResistanceFactor', mu)):
        _put(hygro, tag, repr(value))
    _table(hygro, 'MoistureStorageFunction',
           ((0.0, 0.0), (0.5, storage / 2), (1.0, storage)))
    if liquid is not None:
        _table(hygro, 'LiquidTransportationCoefficientSuction',
               ((0.0, liquid), (storage, liquid)))
    if conductivity_water is not None:
        _table(hygro, 'ThermalConductivityMoistureDependent',
               conductivity_water)
    if conductivity_temperature is not None:
        _table(hygro, 'ThermalConductivityTemperatureDependent',
               conductivity_temperature)

    # Explicit constant type records avoid coupling these fixtures to the
    # production serializer or the ticket-06 helper implementation.
    types = ET.Element('BoundaryConditionsType')
    _put(types, 'Version', '1')
    for label, drive in ((2, left), (3, right)):
        boundary = next(item for item in model.findall('Boundaries/Boundary')
                        if int(item.findtext('ID')) == label)
        data = ET.SubElement(boundary, 'TransientData')
        identity = 'ticket09-physics-boundary-' + str(label)
        _put(data, 'UUID', identity)
        _put(data, 'TimestepFileName', '')
        record = ET.SubElement(types, 'BoundaryConditionType')
        mode = drive['mode']
        if mode not in ('pin', 'robin'):
            raise ValueError('Unknown benchmark boundary mode: ' + mode)
        fields = (
            ('UUID', identity), ('Name', 'Ticket09 boundary ' + str(label)),
            ('Protected', 'false'), ('bcType', 'SteadyState'),
            ('bcModel', 'Dirichlet' if mode == 'pin' else 'Neumann'),
            ('UseHeatFlux', 'false'),
            ('UseTemperature', str(mode == 'pin').lower()),
            ('UseHumidity', str(drive.get('humidity') is not None).lower()),
            ('Temperature', drive['temperature']), ('Color', '0x08c4cd'))
        for tag, value in fields:
            _put(record, tag, value)
        if drive.get('humidity') is not None:
            _put(record, 'Humidity', drive['humidity'])
        if mode == 'robin':
            convection = ET.SubElement(record, 'Convection')
            _put(convection, 'Model', 'Fixed Convection Coefficient')
            _put(convection, 'Temperature', drive['temperature'])
            _put(convection, 'FilmCoefficient', drive['film'])
    mesh = ET.fromstring(members['Mesh.xml'])
    for node in mesh.findall('Case/Nodes/Node'):
        _put(node, 'x', float(node.findtext('x')) * width / 2)
        _put(node, 'y', float(node.findtext('y')) * height)
    for name, root in (('Model.xml', model), ('Materials.xml', materials),
                       ('TransientTypeBC.xml', types), ('Mesh.xml', mesh)):
        members[name] = ET.tostring(root)
    with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, content in members.items():
            archive.writestr(name, content)
    return target


def frozen_physics_cases():
    """Return the pre-run physical case definitions, without running a solver."""
    dry_air = {'mode': 'robin', 'temperature': 20.0,
               'humidity': 0.1, 'film': 10.0}
    sealed = {'mode': 'robin', 'temperature': 20.0, 'film': 10.0}
    pins = {'left': {'mode': 'pin', 'temperature': 20.0},
            'right': {'mode': 'pin', 'temperature': 40.0},
            'initial_temperature': 30.0, 'duration': 1000000.0,
            'time_step': 2000.0}
    cold = {'mode': 'robin', 'temperature': -2.0, 'film': 10.0}
    freezing = {'width': 0.02, 'storage': 150.0, 'mu': 1e12,
                'initial_temperature': 2.0, 'time_step': 30.0,
                'left': cold, 'right': cold}
    evaporation = {'width': 0.02, 'storage': 20.0, 'left': dry_air,
                   'right': dry_air, 'vapor_heat': True}
    cases = {
        'absorb': {},
        'dry': {'left': dry_air, 'right': dry_air},
        'sealed': {'left': sealed, 'right': sealed},
        'closed-gradient': dict(pins, mu=1.0),
        'liquid': {'liquid': 1e-6},
        'lambda-T': dict(pins, mu=1e12,
                         conductivity_water=((0.0, 1.0), (2.0, 1.0)),
                         conductivity_temperature=((0.0, 0.5), (60.0, 2.0))),
        'lambda-w': dict(pins, mu=1e12,
                         conductivity_water=((0.0, 1.0), (2.0, 2.0)),
                         conductivity_temperature=((0.0, 1.0), (60.0, 1.0))),
        'lambda-control': dict(pins, mu=1e12),
        'fusion': dict(freezing, fusion=True),
        'fusion-control': dict(freezing, fusion=False),
        'evaporation': dict(evaporation, evaporation=True),
        'evaporation-control': dict(evaporation, evaporation=False),
    }
    drive = {'mode': 'robin', 'temperature': 20.0,
             'humidity': 0.9, 'film': 10.0}
    defaults = dict(nx=32, ny=8, width=0.2, height=0.05, storage=2.0,
                    mu=1.0, density=600.0, heat_capacity=850.0,
                    initial_temperature=20.0, initial_humidity=0.5,
                    time_step=60.0, duration=21600.0,
                    liquid=None, conductivity_water=None,
                    conductivity_temperature=None, fusion=False,
                    evaporation=False, vapor_heat=False)
    return {name: {**defaults, 'left': dict(drive), 'right': dict(drive), **spec}
            for name, spec in cases.items()}
