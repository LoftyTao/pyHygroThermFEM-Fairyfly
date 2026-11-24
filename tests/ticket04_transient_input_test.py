"""Transient input members: TransientTypeBC.xml and timestep input files.

Contract-level tests for the pure-Python reader/writer in
``pyhygrothermfem_fairyfly.transient_input``. Schema mirrors THERM 8 (contract section
4.7); field order and enumeration values follow
``upstream/LibraryFEMTherm/src/Schemas`` and the C++ serializers. Column and
naming details not yet confirmed against a real THERM 8.1 export are
recorded as ``unverified`` in the ticket 04 report (ticket 07 holds them).
"""
import io
import zipfile
import xml.etree.ElementTree as ET

import pytest

from pyhygrothermfem_fairyfly.errors import ContractError
from pyhygrothermfem_fairyfly.transient_input import (
    TIMESTEP_DIR,
    TimestepFile,
    load_timestep_file,
    load_transient_type_bc,
    save_timestep_file,
    save_transient_type_bc,
    transient_timestep_member,
    validate_transient_inputs,
)


# --- fixtures ---------------------------------------------------------------

def _type_record(uuid='b1-uuid', name='Outside air', bc_type='Transient',
                 bc_model='Neumann', convection='ASHRAE/NFRC Outside',
                 use_temperature=False, use_humidity=False,
                 temperature=None, humidity=None, radiation=None,
                 solar=None, use_heat_flux=False, heat_flux=None,
                 protected=True, color='0x0000FF'):
    """Build a TransientTypeBC record dict."""
    record = {
        'uuid': uuid, 'name': name, 'protected': protected,
        'bc_type': bc_type, 'bc_model': bc_model,
        'convection': (
            None if convection is None else
            convection if isinstance(convection, dict)
            else {'model': convection}),
        'radiation': radiation, 'solar': solar,
        'use_heat_flux': use_heat_flux, 'heat_flux': heat_flux,
        'use_temperature': use_temperature, 'temperature': temperature,
        'use_humidity': use_humidity, 'humidity': humidity,
        'color': color,
    }
    return record


def _model_xml(boundaries=()):
    """Minimal Model.xml fragment hosting Boundary/TransientData elements."""
    parts = ['<ThermModel><Version>1</Version><Boundaries>']
    for boundary in boundaries:
        parts.append('<Boundary><ID>{}</ID><Name>{}</Name>'
                     .format(boundary.get('id', '1'), boundary.get('name', 'b')))
        if 'transient' in boundary:
            transient = boundary['transient']
            if transient.get('omit_file'):
                file_field = ''
            else:
                file_field = ('<TimestepFileName>{}</TimestepFileName>'
                              .format(transient.get('file', '')))
            parts.append('<TransientData><UUID>{}</UUID>'
                         '{}'
                         '</TransientData>'.format(
                             transient.get('uuid', ''), file_field))
        parts.append('</Boundary>')
    parts.append('</Boundaries></ThermModel>')
    return ''.join(parts)


# --- TransientTypeBC.xml ----------------------------------------------------

def test_transient_type_bc_roundtrip():
    """Sample THERM 8 layout round-trips with zero information loss."""
    records = [
        _type_record(),
        _type_record(uuid='ss-uuid', name='SteadyState Interior',
                     bc_type='SteadyState',
                     convection={'model': 'Fixed Convection Coefficient',
                                 'temperature': 22.0, 'film_coefficient': 2.5},
                     radiation={'model': 'Automatic Enclosure',
                                'temperature': 10.0, 'surface_emissivity': 0.9,
                                'view_factor': 1.0},
                     color='0x00FF00'),
        _type_record(uuid='d-uuid', name='Fixed temperature and humidity',
                     bc_model='Dirichlet', use_temperature=True,
                     temperature=20.0, use_humidity=True, humidity=0.5,
                     protected=False, color='0xA6CAF0'),
        _type_record(uuid='hf-uuid', name='Constant Heat Flux',
                     use_heat_flux=True, heat_flux=3.5),
    ]
    xml = save_transient_type_bc(records)
    root = ET.fromstring(xml)
    assert root.tag == 'BoundaryConditionsType'
    assert root.findtext('Version') == '1'
    parsed = load_transient_type_bc(xml)
    assert parsed == records
    # Canonical record field order mirrors the THERM 8 sample file.
    fields = [child.tag for child in root.findall('BoundaryConditionType')[1]]
    assert fields == ['UUID', 'Name', 'Protected', 'bcType', 'bcModel',
                      'Convection', 'Radiation', 'UseHeatFlux',
                      'UseTemperature', 'UseHumidity', 'Color']
    convection = [child.tag for child in root.findall(
        'BoundaryConditionType')[1].find('Convection')]
    assert convection == ['Model', 'Temperature', 'FilmCoefficient']


def test_transient_type_bc_reads_reference_sample():
    """The upstream schema sample parses field for field."""
    text = (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        '<BoundaryConditionsType><Version>1</Version>'
        '<BoundaryConditionType>'
        '<UUID>8a0494b0-d5ba-11ea-87d0-0242ac130003</UUID>'
        '<Name>Fixed film coefficient - Indoor</Name>'
        '<Protected>true</Protected><bcType>Transient</bcType>'
        '<bcModel>Neumann</bcModel>'
        '<Convection><Model>Fixed Convection Coefficient</Model></Convection>'
        '<UseHeatFlux>false</UseHeatFlux><UseTemperature>false</UseTemperature>'
        '<UseHumidity>false</UseHumidity><Color>0xFF0000</Color>'
        '</BoundaryConditionType>'
        '<BoundaryConditionType>'
        '<UUID>ss</UUID><Name>SteadyState Interior</Name>'
        '<Protected>true</Protected><bcType>SteadyState</bcType>'
        '<bcModel>Neumann</bcModel>'
        '<Convection><Model>Fixed Convection Coefficient</Model>'
        '<Temperature>22</Temperature><FilmCoefficient>2.5</FilmCoefficient>'
        '</Convection>'
        '<Radiation><Model>Automatic Enclosure</Model><Temperature>10'
        '</Temperature><SurfaceEmissivity>0.9</SurfaceEmissivity>'
        '<ViewFactor>1</ViewFactor></Radiation>'
        '<UseHeatFlux>false</UseHeatFlux><UseTemperature>false</UseTemperature>'
        '<UseHumidity>false</UseHumidity><Color>0x00FF00</Color>'
        '</BoundaryConditionType></BoundaryConditionsType>')
    records = load_transient_type_bc(text)
    assert len(records) == 2
    assert records[0]['convection'] == {'model': 'Fixed Convection Coefficient'}
    assert records[1]['convection']['film_coefficient'] == 2.5
    assert records[1]['radiation']['view_factor'] == 1.0
    assert records[1]['use_humidity'] is False


def test_transient_type_bc_missing_version_defaults_to_one():
    """An absent Version is tolerated and treated as version 1.

    The official THERM 8.1 sample input
    (``Stucco Wall - Moisture.thmz``, ``CreationVersion 8.0.7.0``) ships a
    ``TransientTypeBC.xml`` whose ``BoundaryConditionsType`` root carries no
    ``Version`` element even though the XSD marks it ``minOccurs=1``; the
    upstream C++ reader never reads the field. The reader must accept the
    genuine absence while still refusing present-but-malformed values.
    """
    assert load_transient_type_bc('<BoundaryConditionsType/>') == []
    records = load_transient_type_bc(
        '<BoundaryConditionsType>'
        '<BoundaryConditionType>'
        '<UUID>8a0494b0-d5ba-11ea-87d0-0242ac130003</UUID>'
        '<Name>Fixed film coefficient - Indoor</Name>'
        '<Protected>true</Protected><bcType>Transient</bcType>'
        '<bcModel>Neumann</bcModel>'
        '<Convection><Model>Fixed Convection Coefficient</Model></Convection>'
        '<UseHeatFlux>false</UseHeatFlux><UseTemperature>false</UseTemperature>'
        '<UseHumidity>false</UseHumidity><Color>0xFF0000</Color>'
        '</BoundaryConditionType></BoundaryConditionsType>')
    assert len(records) == 1
    assert records[0]['name'] == 'Fixed film coefficient - Indoor'
    assert records[0]['convection'] == {'model': 'Fixed Convection Coefficient'}


def test_transient_type_bc_rejects_unknown_or_malformed_version():
    """A present Version must be '1'; anything else is refused."""
    for version in ('2', 'x', ''):
        with pytest.raises(ValueError, match='THMZ-E20'):
            load_transient_type_bc(
                '<BoundaryConditionsType><Version>{}</Version>'
                '</BoundaryConditionsType>'.format(version))


def test_transient_type_bc_rejects_contract_violations():
    """Missing fields, bad enums and unknown tags carry THMZ-E codes."""
    with pytest.raises(ValueError, match='THMZ-E20'):
        load_transient_type_bc('<Wrong/>')
    with pytest.raises(ValueError, match='THMZ-E20'):
        load_transient_type_bc(
            '<BoundaryConditionsType><Version>1</Version>'
            '<BoundaryConditionType><UUID>u</UUID></BoundaryConditionType>'
            '</BoundaryConditionsType>')
    with pytest.raises(ValueError, match='THMZ-E20'):
        load_transient_type_bc(
            '<BoundaryConditionsType><Version>1</Version>'
            '<BoundaryConditionType><UUID>u</UUID><Name>n</Name>'
            '<Protected>true</Protected><bcType>Other</bcType>'
            '<bcModel>Neumann</bcModel><UseHeatFlux>false</UseHeatFlux>'
            '<UseTemperature>false</UseTemperature><UseHumidity>false'
            '</UseHumidity><Color>c</Color></BoundaryConditionType>'
            '</BoundaryConditionsType>')
    with pytest.raises(ValueError, match='THMZ-E20'):
        load_transient_type_bc(
            '<BoundaryConditionsType><Version>1</Version>'
            '<BoundaryConditionType><UUID>u</UUID><Name>n</Name>'
            '<Protected>true</Protected><bcType>Transient</bcType>'
            '<bcModel>Neumann</bcModel><Bogus>x</Bogus>'
            '<UseHeatFlux>false</UseHeatFlux><UseTemperature>false'
            '</UseTemperature><UseHumidity>false</UseHumidity>'
            '<Color>c</Color></BoundaryConditionType>'
            '</BoundaryConditionsType>')
    with pytest.raises(ValueError, match='THMZ-E20'):
        load_transient_type_bc(
            '<BoundaryConditionsType><Version>1</Version>'
            '<BoundaryConditionType><UUID>u</UUID><Name>n</Name>'
            '<Protected>true</Protected><bcType>Transient</bcType>'
            '<bcModel>Neumann</bcModel>'
            '<Convection><Model>ASHRAE/NFRC Outside</Model></Convection>'
            '<UseHeatFlux>false</UseHeatFlux><UseTemperature>false'
            '</UseTemperature><UseHumidity>false</UseHumidity>'
            '<Color>c</Color></BoundaryConditionType>'
            '<BoundaryConditionType><UUID>u</UUID><Name>n2</Name>'
            '<Protected>true</Protected><bcType>Transient</bcType>'
            '<bcModel>Neumann</bcModel><UseHeatFlux>false</UseHeatFlux>'
            '<UseTemperature>false</UseTemperature><UseHumidity>false'
            '</UseHumidity><Color>c</Color></BoundaryConditionType>'
            '</BoundaryConditionsType>')


# --- timestep input files ---------------------------------------------------

def test_timestep_file_roundtrip_fixed_film():
    """FixedConvectionFilmTimestep serializes in serializer field order."""
    content = TimestepFile(
        name='Exterior',
        convection={'FixedConvectionFilmTimestep': [
            {'index': 1, 'temperature': 40.0, 'humidity': 0.3,
             'film_coefficient': 10.0},
            {'index': 2, 'temperature': 38.5, 'humidity': 0.35,
             'film_coefficient': 10.0}],
        },
        temperature=(),
    )
    xml = save_timestep_file(content)
    root = ET.fromstring(xml)
    assert root.tag == 'InputBoundaryConditionsData'
    assert root.findtext('Version') == '1'
    assert root.findtext('Name') == 'Exterior'
    steps = root.findall(
        'BoundaryConditionTransient/ConvectionTimesteps/'
        'FixedConvectionFilmTimestep')
    assert [child.tag for child in steps[0]] == [
        'Index', 'FilmCoefficient', 'Temperature', 'Humidity']
    parsed = load_timestep_file(xml)
    assert parsed == content


def test_timestep_file_roundtrip_scalar_series():
    """Temperature/humidity/flux/solar series round-trip without Index."""
    content = TimestepFile(
        name='Fixed T RH',
        temperature=({'index': None, 'temperature': 20.0},
                     {'index': None, 'temperature': 21.0}),
        humidity=({'index': None, 'humidity': 0.0},) * 2,
        heat_flux=({'index': 1, 'heat_flux': -2.5},),
        solar_radiation=({'index': 1, 'solar_radiation': 800.0},),
    )
    parsed = load_timestep_file(save_timestep_file(content))
    assert parsed == content


def test_timestep_file_radiation_series_roundtrip():
    """Radiation timestep records keep their model-specific fields."""
    content = TimestepFile(
        name='Rad',
        radiation={'FixedRadiationCoefficientTimestep': [
            {'index': 1, 'temperature': 5.0, 'hr': 4.0}],
            'BlackBodyRadiationTimestep': [
            {'index': 1, 'temperature': 5.0, 'emissivity': 0.9}]},
    )
    xml = save_timestep_file(content)
    parsed = load_timestep_file(xml)
    assert parsed == content


def test_timestep_file_accepts_coefficient_tag_alias():
    """Reader takes both spellings of the fixed radiation timestep tag."""
    xml = (
        '<InputBoundaryConditionsData><Version>1</Version><Name>r</Name>'
        '<BoundaryConditionTransient><RadiationTimesteps>'
        '<FixedRadiationTimestep><Index>1</Index><Temperature>5</Temperature>'
        '<hr>4</hr></FixedRadiationTimestep>'
        '<FixedRadiationCoefficientTimestep><Index>2</Index>'
        '<Temperature>6</Temperature><hr>4</hr>'
        '</FixedRadiationCoefficientTimestep>'
        '</RadiationTimesteps></BoundaryConditionTransient>'
        '</InputBoundaryConditionsData>')
    parsed = load_timestep_file(xml)
    assert [s['temperature'] for s in
            parsed.radiation['FixedRadiationCoefficientTimestep']] == [5.0, 6.0]


def test_timestep_file_missing_version_defaults_to_one():
    """An absent Version is tolerated and treated as version 1.

    Same upstream-lenient rule as ``TransientTypeBC.xml``: the C++ timestep
    reader (``BCTransient/Timestep.cxx``) treats ``Version`` as an optional
    scalar, so only present-but-unrecognized values are contract violations.
    """
    parsed = load_timestep_file(
        '<InputBoundaryConditionsData><Name>x</Name>'
        '<BoundaryConditionTransient><TemperatureTimestep>'
        '<Temperature>5</Temperature></TemperatureTimestep>'
        '</BoundaryConditionTransient></InputBoundaryConditionsData>')
    assert parsed.name == 'x'
    assert [e['temperature'] for e in parsed.temperature] == [5.0]


def test_timestep_file_rejects_unknown_or_malformed_version():
    """A present Version must be '1'; anything else is refused."""
    for version in ('2', 'x', ''):
        with pytest.raises(ValueError, match='THMZ-E20'):
            load_timestep_file(
                '<InputBoundaryConditionsData><Version>{}</Version>'
                '<Name>x</Name></InputBoundaryConditionsData>'
                .format(version))


def test_timestep_file_rejects_contract_violations():
    """Bad indices, missing fields and humidity out of range raise E2 codes."""
    with pytest.raises(ValueError, match='THMZ-E20'):
        load_timestep_file(
            '<InputBoundaryConditionsData><Version>1</Version><Name>x</Name>'
            '<BoundaryConditionTransient><TemperatureTimestep>'
            '<Index>0</Index><Temperature>1</Temperature>'
            '</TemperatureTimestep></BoundaryConditionTransient>'
            '</InputBoundaryConditionsData>')
    with pytest.raises(ValueError, match='THMZ-E20'):
        load_timestep_file(
            '<InputBoundaryConditionsData><Version>1</Version><Name>x</Name>'
            '<BoundaryConditionTransient><HumidityTimestep>'
            '<Humidity>1.5</Humidity></HumidityTimestep>'
            '</BoundaryConditionTransient></InputBoundaryConditionsData>')
    with pytest.raises(ValueError, match='THMZ-E20'):
        load_timestep_file(
            '<InputBoundaryConditionsData><Version>1</Version><Name>x</Name>'
            '<BoundaryConditionTransient><ConvectionTimesteps>'
            '<TARPTimestep><Temperature>20</Temperature></TARPTimestep>'
            '</ConvectionTimesteps></BoundaryConditionTransient>'
            '</InputBoundaryConditionsData>')
    with pytest.raises(ValueError, match='THMZ-E20'):
        load_timestep_file(
            '<InputBoundaryConditionsData><Version>1</Version><Name>x</Name>'
            '<BoundaryConditionTransient><MysteryTimestep><Index>1</Index>'
            '<Temperature>20</Temperature></MysteryTimestep>'
            '</BoundaryConditionTransient></InputBoundaryConditionsData>')


def test_timestep_member_name_helper():
    """Member names join the fixed timestep directory with the file name."""
    assert transient_timestep_member('out.xml') == \
        'timestep input files/out.xml'
    assert TIMESTEP_DIR == 'timestep input files'


# --- cross-member validation ------------------------------------------------

def _archive_members(records=(), timestep_files=(), model_xml=None):
    members = {}
    members['TransientTypeBC.xml'] = save_transient_type_bc(records).encode()
    for name, content in timestep_files.items():
        members[TIMESTEP_DIR + '/' + name] = content.encode()
    if model_xml is not None:
        members['Model.xml'] = model_xml.encode()
    return members


def test_validate_transient_inputs_happy_path():
    records = [_type_record(uuid='bc-1',
                            convection='Fixed Convection Coefficient')]
    members = _archive_members(
        records, {'out.xml': save_timestep_file(
            TimestepFile(name='Outside air',
                         convection={'FixedConvectionFilmTimestep': [
                            {'index': 1, 'temperature': 40.0,
                             'humidity': 0.3, 'film_coefficient': 10.0}]},
                         temperature=({'index': None, 'temperature': 20.0},)))},
        _model_xml([{'transient': {'uuid': 'bc-1', 'file': 'out.xml'}}]))
    references = validate_transient_inputs(members)
    assert len(references) == 1
    assert references[0]['boundary_id'] == '1'
    assert references[0]['timestep_member'] == 'timestep input files/out.xml'
    assert references[0]['record']['uuid'] == 'bc-1'


def test_validate_transient_inputs_missing_file_and_uuid():
    members = _archive_members(
        [], {}, _model_xml([{'transient': {'uuid': 'ghost',
                                          'file': 'missing.xml'}}]))
    with pytest.raises(ValueError, match='THMZ-E40|THMZ-E20'):
        validate_transient_inputs(members)
    # Unreferenced timestep member violates consistency (E5013).
    members = _archive_members(
        [], {'orphan.xml': save_timestep_file(
            TimestepFile(name='x',
                         temperature=({'index': None, 'temperature': 1.0},)))},
        _model_xml())
    with pytest.raises(ValueError, match='THMZ-E5013'):
        validate_transient_inputs(members)


def test_validate_transient_inputs_montazeri_rejected():
    """Montazeri declares at the data layer but has no kernel support."""
    records = [_type_record(uuid='m', convection='Montazeri')]
    members = _archive_members(
        records,
        {'m.xml': save_timestep_file(TimestepFile(
            name='Montazeri',
            convection={'MontazeriTimestep': [
                {'index': 1, 'temperature': 20.0, 'humidity': 0.4,
                 'wind_speed': 3.0, 'wind_direction': 180.0}]}))},
        _model_xml([{'transient': {'uuid': 'm', 'file': 'm.xml'}}]))
    with pytest.raises(ValueError, match='THMZ-E5001'):
        validate_transient_inputs(members)


def test_validate_steady_state_record_needs_no_timestep_file():
    """bcType=SteadyState TransientData carries an empty/absent file name.

    The fairyfly-next-therm writer emits ``<TimestepFileName/>`` (empty) for
    steady-type records; both the empty element and an omitted element must
    validate, and per-series checks must not run on the constant record.
    """
    record = _type_record(uuid='ss', name='Steady bc',
                          bc_type='SteadyState', bc_model='Dirichlet',
                          convection={'model': 'Fixed Convection Coefficient',
                                      'temperature': 21.0,
                                      'film_coefficient': 2.5},
                          use_temperature=True, temperature=21.0,
                          use_humidity=True, humidity=0.5,
                          use_heat_flux=True, heat_flux=-3.0)
    for transient in ({'uuid': 'ss', 'file': ''},
                      {'uuid': 'ss', 'omit_file': True}):
        members = _archive_members(
            [record], {}, _model_xml([{'transient': transient}]))
        references = validate_transient_inputs(members)
        assert len(references) == 1
        assert references[0]['timestep_file'] is None
        assert references[0]['timestep_member'] is None


def test_validate_transient_record_requires_timestep_file():
    """bcType=Transient still requires a resolvable timestep member."""
    record = _type_record(uuid='tr')
    members = _archive_members(
        [record], {}, _model_xml([{'transient': {'uuid': 'tr', 'file': ''}}]))
    with pytest.raises(ValueError, match='THMZ-E20|THMZ-E40'):
        validate_transient_inputs(members)


def test_timestep_file_name_in_zip_member():
    """Model.xml TimestepFileName resolves against the fixed directory."""
    records = [_type_record(uuid='bc-1',
                            convection='Fixed Convection Coefficient')]
    model_xml = _model_xml(
        [{'transient': {'uuid': 'bc-1', 'file': 'timestep input files/x.xml'}}])
    members = _archive_members(
        records, {'x.xml': save_timestep_file(
            TimestepFile(name='Outside air',
                         convection={'FixedConvectionFilmTimestep': [
                            {'index': 1, 'temperature': 20.0, 'humidity': 0.4,
                             'film_coefficient': 5.0}]}))},
        model_xml)
    references = validate_transient_inputs(members)
    assert references[0]['timestep_member'] == 'timestep input files/x.xml'


@pytest.mark.parametrize('content, code, value', [
    ('<InputBoundaryConditionsData>', 'THMZ-E1007', None),
    ('<InputBoundaryConditionsData><Mystery/>', 'THMZ-E1007', None),
    ('<InputBoundaryConditionsData><Name>x</Name><Mystery/>'
     '</InputBoundaryConditionsData>', 'THMZ-E2001', 'Mystery'),
    ('<InputBoundaryConditionsData/>', 'THMZ-E2003', 'Name'),
    ('<InputBoundaryConditionsData><Name>x</Name>'
     '<BoundaryConditionTransient><TemperatureTimestep>'
     '<Temperature>invalid</Temperature></TemperatureTimestep>'
     '</BoundaryConditionTransient></InputBoundaryConditionsData>',
     'THMZ-E2004', 'invalid'),
    ('<InputBoundaryConditionsData><Name>x</Name>'
     '<BoundaryConditionTransient><HumidityTimestep>'
     '<Humidity>1.5</Humidity></HumidityTimestep>'
     '</BoundaryConditionTransient></InputBoundaryConditionsData>',
     'THMZ-E2004', '1.5'),
])
def test_input_reader_uses_contract_error_codes(content, code, value):
    """Input failures expose the specific code and invalid field value."""
    with pytest.raises(ContractError) as caught:
        load_timestep_file(content)
    assert caught.value.code == code
    if value is not None:
        assert value in str(caught.value)


@pytest.mark.parametrize('field, value', [
    ('bcType', 'Other'), ('Protected', 'maybe'), ('Protected', 'MAYBE'),
])
def test_type_reader_identifies_invalid_enumerations(field, value):
    """Illegal type and Boolean fields report E2005 and the actual value."""
    root = ET.fromstring(save_transient_type_bc([_type_record()]))
    root.find('BoundaryConditionType/' + field).text = value
    with pytest.raises(ContractError) as caught:
        load_transient_type_bc(ET.tostring(root))
    assert caught.value.code == 'THMZ-E2005'
    assert field in str(caught.value)
    assert value in str(caught.value)


@pytest.mark.parametrize('loader, content, field', [
    (load_transient_type_bc,
     '<BoundaryConditionsType><Version>1</Version><Version>2</Version>'
     '</BoundaryConditionsType>', 'Version'),
    (load_timestep_file,
     '<InputBoundaryConditionsData><Version>1</Version><Version>2</Version>'
     '<Name>x</Name></InputBoundaryConditionsData>', 'Version'),
    (load_timestep_file,
     '<InputBoundaryConditionsData><Name>x</Name><Name>y</Name>'
     '</InputBoundaryConditionsData>', 'Name'),
    (load_timestep_file,
     '<InputBoundaryConditionsData><Name>x</Name>'
     '<BoundaryConditionTransient/><BoundaryConditionTransient/>'
     '</InputBoundaryConditionsData>', 'BoundaryConditionTransient'),
    (load_timestep_file,
     '<InputBoundaryConditionsData><Name>x</Name>'
     '<BoundaryConditionTransient><Index>1</Index><Index>2</Index>'
     '</BoundaryConditionTransient></InputBoundaryConditionsData>', 'Index'),
    (load_timestep_file,
     '<InputBoundaryConditionsData><Name>x</Name>'
     '<BoundaryConditionTransient><ConvectionTimesteps/>'
     '<ConvectionTimesteps/></BoundaryConditionTransient>'
     '</InputBoundaryConditionsData>', 'ConvectionTimesteps'),
    (load_timestep_file,
     '<InputBoundaryConditionsData><Name>x</Name>'
     '<BoundaryConditionTransient><RadiationTimesteps/>'
     '<RadiationTimesteps/></BoundaryConditionTransient>'
     '</InputBoundaryConditionsData>', 'RadiationTimesteps'),
])
def test_input_reader_rejects_duplicate_singletons(loader, content, field):
    """Repeated singleton fields are errors, not first-value selection."""
    with pytest.raises(ContractError) as caught:
        loader(content)
    assert caught.value.code == 'THMZ-E2002'
    assert field in str(caught.value)


@pytest.mark.parametrize('key, value', [('uuid', 'bc-1'), ('name', 'Outside')])
def test_type_reader_reports_duplicate_record_identity(key, value):
    """Duplicate record identities carry E2002 and the repeated identity."""
    first = _type_record(uuid='bc-1', name='Outside')
    second = _type_record(uuid='bc-2', name='Inside')
    second[key] = value
    root = ET.fromstring(save_transient_type_bc([first]))
    root.append(ET.fromstring(save_transient_type_bc([second])).find(
        'BoundaryConditionType'))
    with pytest.raises(ContractError) as caught:
        load_transient_type_bc(ET.tostring(root))
    assert caught.value.code == 'THMZ-E2002'
    assert value in str(caught.value)


def test_validator_rejects_multiple_transient_data_on_one_boundary():
    """One boundary cannot silently choose between transient references."""
    root = ET.fromstring(_model_xml(
        [{'transient': {'uuid': 'bc-1', 'file': 'out.xml'}}]))
    boundary = root.find('Boundaries/Boundary')
    ET.SubElement(boundary, 'TransientData')
    members = _archive_members([_type_record(uuid='bc-1')], {},
                               ET.tostring(root, encoding='unicode'))
    with pytest.raises(ContractError) as caught:
        validate_transient_inputs(members)
    assert caught.value.code == 'THMZ-E2002'
    assert 'TransientData' in str(caught.value)


@pytest.mark.parametrize('series', ['convection', 'radiation'])
def test_timestep_writer_rejects_unknown_series(series):
    """An unknown series cannot disappear from a successful write."""
    content = TimestepFile(name='x', **{series: {'MysteryTimestep': [
        {'index': 1, 'temperature': 20.0}]}})
    with pytest.raises(ContractError) as caught:
        save_timestep_file(content)
    assert caught.value.code == 'THMZ-E2001'
    assert 'MysteryTimestep' in str(caught.value)


def test_timestep_writer_rejects_unknown_entry_fields():
    """Extra entry fields are refused instead of losing their values."""
    content = TimestepFile(name='x', temperature=(
        {'index': 1, 'temperature': 20.0, 'extra': 9.0},))
    with pytest.raises(ContractError) as caught:
        save_timestep_file(content)
    assert caught.value.code == 'THMZ-E2001'
    assert 'extra' in str(caught.value)


@pytest.mark.parametrize('subrecord', [None, 'convection', 'radiation', 'solar'])
def test_type_writer_rejects_unknown_record_fields(subrecord):
    """Unknown type-record values must not be silently omitted."""
    record = _type_record(
        convection={'model': 'Fixed Convection Coefficient'},
        radiation={'model': 'Black Body Radiation'},
        solar={'solar_radiation': 5.0})
    target = record if subrecord is None else record[subrecord]
    target['extra'] = 9.0
    with pytest.raises(ContractError) as caught:
        save_transient_type_bc([record])
    assert caught.value.code == 'THMZ-E2001'
    assert 'extra' in str(caught.value)


@pytest.mark.parametrize('value', ['invalid', None, float('nan'),
                                  float('inf'), True, -274.0])
def test_timestep_writer_reports_invalid_numeric_values(value):
    """Invalid written temperatures produce E2004, not native exceptions."""
    content = TimestepFile(name='x', temperature=(
        {'index': 1, 'temperature': value},))
    with pytest.raises(ContractError) as caught:
        save_timestep_file(content)
    assert caught.value.code == 'THMZ-E2004'
    assert 'Temperature' in str(caught.value)
    assert repr(value) in str(caught.value)


@pytest.mark.parametrize('value', [0, -1, 1.5, True, 'invalid'])
@pytest.mark.parametrize('location', ['file', 'entry'])
def test_timestep_writer_rejects_non_positive_integer_indices(value, location):
    """Index values are never truncated or coerced from Boolean values."""
    content = TimestepFile(name='x', temperature=(
        {'index': None, 'temperature': 20.0},))
    if location == 'file':
        content.index = value
    else:
        content.temperature[0]['index'] = value
    with pytest.raises(ContractError) as caught:
        save_timestep_file(content)
    assert caught.value.code == 'THMZ-E2004'
    assert repr(value) in str(caught.value)


@pytest.mark.parametrize('field', ['uuid', 'name', 'protected', 'bc_type',
                                  'bc_model', 'color', 'use_heat_flux',
                                  'use_temperature', 'use_humidity'])
def test_type_writer_reports_missing_required_fields(field):
    """Missing type fields carry E2003 instead of leaking KeyError."""
    record = _type_record()
    del record[field]
    with pytest.raises(ContractError) as caught:
        save_transient_type_bc([record])
    assert caught.value.code == 'THMZ-E2003'
    assert field in str(caught.value)


def test_timestep_writer_reports_missing_required_entry_field():
    """Missing series values carry the missing-field code E2003."""
    content = TimestepFile(name='x', temperature=({'index': 1},))
    with pytest.raises(ContractError) as caught:
        save_timestep_file(content)
    assert caught.value.code == 'THMZ-E2003'
    assert 'temperature' in str(caught.value)


@pytest.mark.parametrize('field', ['protected', 'use_heat_flux',
                                  'use_temperature', 'use_humidity'])
def test_type_writer_does_not_coerce_boolean_strings(field):
    """The string 'false' cannot silently become a true declaration."""
    record = _type_record()
    record[field] = 'false'
    with pytest.raises(ContractError) as caught:
        save_transient_type_bc([record])
    assert caught.value.code == 'THMZ-E2005'
    assert field in str(caught.value)
    assert 'false' in str(caught.value)


def test_type_writer_rejects_invalid_humidity_value():
    """The write-side humidity range matches the reader's range."""
    with pytest.raises(ContractError) as caught:
        save_transient_type_bc([_type_record(humidity=1.5)])
    assert caught.value.code == 'THMZ-E2004'
    assert '1.5' in str(caught.value)


@pytest.mark.parametrize('value', ['0', '-1', '1.5', 'invalid'])
def test_timestep_reader_checks_file_index_range(value):
    """The file-level Index follows the same one-based rule as entries."""
    content = ('<InputBoundaryConditionsData><Name>x</Name>'
               '<BoundaryConditionTransient><Index>{}</Index>'
               '</BoundaryConditionTransient></InputBoundaryConditionsData>'
               .format(value))
    with pytest.raises(ContractError) as caught:
        load_timestep_file(content)
    assert caught.value.code == 'THMZ-E2004'
    assert value in str(caught.value)


def test_timestep_roundtrip_preserves_schema_location():
    """The official input's schema-location hint survives XML round trips."""
    content = (
        '<InputBoundaryConditionsData '
        'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
        'xsi:noNamespaceSchemaLocation="BoundaryConditionsTimestep.xsd">'
        '<Name>Official input</Name><BoundaryConditionTransient/>'
        '</InputBoundaryConditionsData>')
    parsed = load_timestep_file(content)
    written = ET.fromstring(save_timestep_file(parsed))
    assert written.attrib == {
        '{http://www.w3.org/2001/XMLSchema-instance}noNamespaceSchemaLocation':
        'BoundaryConditionsTimestep.xsd'}
    assert load_timestep_file(ET.tostring(written)) == parsed


@pytest.mark.parametrize('loader, content', [
    (load_transient_type_bc, '<BoundaryConditionsType extra="9"/>'),
    (load_timestep_file,
     '<InputBoundaryConditionsData extra="9"><Name>x</Name>'
     '</InputBoundaryConditionsData>'),
    (load_timestep_file,
     '<InputBoundaryConditionsData><Name extra="9">x</Name>'
     '</InputBoundaryConditionsData>'),
    (load_timestep_file,
     '<InputBoundaryConditionsData><Name>x<extra>9</extra></Name>'
     '</InputBoundaryConditionsData>'),
])
def test_input_reader_rejects_unrepresentable_xml_values(loader, content):
    """Unrecognized XML values cannot disappear during a round trip."""
    with pytest.raises(ContractError) as caught:
        loader(content)
    assert caught.value.code == 'THMZ-E2001'
    assert 'extra' in str(caught.value)


@pytest.mark.parametrize('missing', [
    'Model.xml', 'TransientTypeBC.xml', 'timestep input files/out.xml',
])
def test_validator_reports_missing_required_archive_members(missing):
    """Absent input members produce the archive-specific E1004 code."""
    members = _archive_members(
        [_type_record(uuid='bc-1',
                      convection='Fixed Convection Coefficient')],
        {'out.xml': save_timestep_file(TimestepFile(
            name='Outside air', convection={'FixedConvectionFilmTimestep': (
                {'index': 1, 'film_coefficient': 10.0, 'temperature': 20.0,
                 'humidity': 0.5},)}))},
        _model_xml([{'transient': {'uuid': 'bc-1', 'file': 'out.xml'}}]))
    del members[missing]
    with pytest.raises(ContractError) as caught:
        validate_transient_inputs(members)
    assert caught.value.code == 'THMZ-E1004'
    assert missing in str(caught.value)


def test_validator_requires_type_member_without_transient_references():
    """A transient archive still needs its type table when it is empty."""
    with pytest.raises(ContractError) as caught:
        validate_transient_inputs({'Model.xml': _model_xml().encode()})
    assert caught.value.code == 'THMZ-E1004'
    assert 'TransientTypeBC.xml' in str(caught.value)
