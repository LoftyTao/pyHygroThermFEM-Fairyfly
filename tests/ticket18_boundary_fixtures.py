"""Explicit constant boundary declarations for legacy transient test cases.

Ticket 18 decision A forbids the former implicit SteadyStateBC fallback.
These fixtures keep their authored physics and thresholds by declaring
constant TransientTypeBC records instead. No production writer is changed.
"""
import xml.etree.ElementTree as ET


def declare_constant_boundaries(members):
    """Migrate legacy fixture records without changing their driving values."""
    model = ET.fromstring(members['Model.xml'])
    if model.findtext('Properties/CalculationOptions/CalculationMode') != 'Transient':
        return
    wet = model.findtext('Properties/CalculationOptions/SimulateMoisture') == 'true'
    bcs = ET.fromstring(members['SteadyStateBC.xml'])
    types = ET.fromstring(members['TransientTypeBC.xml'])
    records = {b.findtext('Name'): b for b in bcs.findall('BoundaryCondition')}

    def put(parent, tag, value):
        ET.SubElement(parent, tag).text = str(value)

    for boundary in model.findall('Boundaries/Boundary'):
        if boundary.find('TransientData') is not None:
            continue
        record = records[boundary.findtext('Name')]
        block = record.find('Simplified')
        simple = block is not None
        if not simple:
            block = record.find('Comprehensive')
        convection = block if simple else block.find('Convection')
        film = float(convection.findtext('FilmCoefficient'))
        flux = float(block.findtext('ConstantFlux/Flux') or '0')
        radiation = block.find('Radiation/BlackBodyRadiation')
        radiant = radiation is not None and float(radiation.findtext('Emissivity')) != 0
        humidity = block.find('RelativeHumidity')
        if film == 0 and flux == 0 and not radiant:
            # Zero film means no moisture exchange, even with legacy RH data.
            continue
        identity = 'ticket18-explicit-' + boundary.findtext('ID')
        data = ET.SubElement(boundary, 'TransientData')
        put(data, 'UUID', identity)
        put(data, 'TimestepFileName', '')
        target = ET.SubElement(types, 'BoundaryConditionType')
        for tag, value in (('UUID', identity), ('Name', identity),
                           ('Protected', 'false'), ('bcType', 'SteadyState'),
                           ('bcModel', 'Neumann')):
            put(target, tag, value)
        drive = ET.SubElement(target, 'Convection')
        put(drive, 'Model', 'Fixed Convection Coefficient')
        put(drive, 'FilmCoefficient', film)
        put(drive, 'Temperature', convection.findtext('Temperature'))
        if radiant:
            raise ValueError('Radiating fixtures require an explicitly authored '
                             'TransientTypeBC; automatic conversion is not equivalent')
        put(target, 'UseHeatFlux', str(flux != 0).lower())
        if flux:
            put(target, 'HeatFlux', flux)
        put(target, 'UseTemperature', 'false')
        humid = wet and film > 0 and humidity is not None
        put(target, 'UseHumidity', str(humid).lower())
        if humid:
            put(target, 'Humidity', humidity.text)
        put(target, 'Color', '0x000000')
    for name, root in (('Model.xml', model), ('SteadyStateBC.xml', bcs),
                       ('TransientTypeBC.xml', types)):
        members[name] = ET.tostring(root)
