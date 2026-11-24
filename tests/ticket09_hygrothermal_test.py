"""Ticket 09: hygrothermal transient dispatch and moisture boundaries.

``slab.thmz`` is patched into a coupled hygrothermal twin: the material
gains the moisture record (density, porosity, heat capacity, mu, sorption)
and ``SimulateMoisture`` flips on. Moisture BC coverage exercises the three
kernel pairs: humid convection (Robin vapor exchange from the convection
timestep's Humidity column), temperature+humidity pins, and humidity-only
pins. Numerical claims stay conservative: exact magnitudes are pending a
real THERM 8.1 transient export (ticket 07, contract section 9).
"""
import math
import zipfile
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

from pyhygrothermfem_fairyfly import (ContractError, SolveError, UnsupportedError,
                          run_thmz, solve)
from pyhygrothermfem_fairyfly import transient_results as series_io
from pyhygrothermfem_fairyfly.transient_input import (
    TIMESTEP_DIR, TimestepFile, save_timestep_file)

from ticket06_transient_dispatch_test import (
    _film_series, _slab_members, _transient_slab, _transient_type_bc,
    HOT_FILE, HOT_UUID, NUM_STEPS, TIME_STEP)

SORPTION = ((0.0, 0.0), (0.5, 75.0), (1.0, 150.0))
INITIAL_HUMIDITY = 0.5


def _wet_hook(members, liquid=False, porosity=True, sorption=True,
              bulk_density=600.0, heat_capacity=850.0,
              sorption_curve=SORPTION, freeze=False, evaporation=False):
    """Turn the patched transient slab into a hygrothermal model."""
    mats = ET.fromstring(members['Materials.xml'])
    hygro = mats.find('Material/Solid/HygroThermal')
    ET.SubElement(hygro, 'BulkDensity').text = repr(bulk_density)
    if porosity:
        ET.SubElement(hygro, 'Porosity').text = '0.2'
    ET.SubElement(hygro, 'SpecificHeatCapacityDry').text = repr(heat_capacity)
    ET.SubElement(hygro, 'WaterVaporDiffusionResistanceFactor').text = '50'
    if sorption:
        storage = ET.SubElement(hygro, 'MoistureStorageFunction')
        for x, y in sorption_curve:
            point = ET.SubElement(storage, 'TableValue')
            ET.SubElement(point, 'x').text = repr(x)
            ET.SubElement(point, 'y').text = repr(y)
    if liquid:
        curve = ET.SubElement(
            hygro, 'LiquidTransportationCoefficientSuction')
        for x, y in ((0.0, 1e-12), (150.0, 1e-8)):
            point = ET.SubElement(curve, 'TableValue')
            ET.SubElement(point, 'x').text = repr(x)
            ET.SubElement(point, 'y').text = repr(y)
    members['Materials.xml'] = ET.tostring(mats)
    model = ET.fromstring(members['Model.xml'])
    model.find(
        'Properties/CalculationOptions/SimulateMoisture').text = 'true'
    modeling = model.find(
        'Properties/CalculationOptions/ModelingOptions')
    if modeling is not None:
        if freeze:
            modeling.find('ExcludeLatentHeatOfFusion').text = 'false'
        if evaporation:
            modeling.find('ExcludeHeatOfEvaporation').text = 'false'
    members['Model.xml'] = ET.tostring(model)


def _wet_slab(tmp_path, name='wet.thmz', **kwargs):
    kwargs.setdefault('density', False)
    hook = kwargs.pop('members_hook', None)
    wet_kwargs = {key: kwargs.pop(key) for key in list(kwargs)
                  if key in ('liquid', 'porosity', 'sorption', 'bulk_density',
                             'heat_capacity', 'sorption_curve', 'freeze',
                             'evaporation')}

    def members_hook(members):
        _wet_hook(members, **wet_kwargs)
        if hook is not None:
            hook(members)

    return _transient_slab(tmp_path, name, members_hook=members_hook,
                           **kwargs)


def _humid_convection(humidity=0.9):
    """Hot Air as a humid Robin boundary (Transient record)."""
    record = _transient_type_bc(use_humidity=True, humidity=0.5)
    timestep = {HOT_FILE: save_timestep_file(TimestepFile(
        name='Hot Air transient',
        convection={'FixedConvectionFilmTimestep': _film_series(
            [20.0], humidity=humidity)}))}
    return record, timestep


def _read(members, name):
    return series_io.read_transient_series(members['transient results/' + name])


# --- coupled solve --------------------------------------------------------

def test_hygrothermal_humid_convection_absorbs(tmp_path):
    """Robin moisture exchange: wet air raises the boundary humidity."""
    record, timestep = _humid_convection(humidity=0.9)
    source = _wet_slab(tmp_path, type_records=[record],
                       timestep_files=timestep)
    result = solve(source)
    assert result.mode == 'hygrothermal-transient'
    assert result.provenance['mode'] == 'hygrothermal-transient'
    assert result.provenance['contract_version'] == 2001
    assert result.steps_completed == NUM_STEPS

    members = result.members
    humidities = _read(members, 'Humidities.csv')
    water = _read(members, 'WaterContent.csv')
    edge_water = _read(members, 'WaterFluxEdges.csv')
    edge_heat = _read(members, 'HeatFluxEdges.csv')

    # Every humidity stays inside the physical [0, 1] band.
    assert all(0.0 <= value <= 1.0
               for row in humidities['values'] for value in row)
    # The wetted (hot) face moves toward the 0.9 driving humidity.
    final_h = humidities['values'][-1]
    assert max(final_h) > INITIAL_HUMIDITY + 1e-4
    # Initial water content follows the sorption curve at phi=0.5 (75 kg/m3)
    # and the slab absorbs water through the wetted face.
    assert water['values'][0] == pytest.approx([75.0] * 9)
    assert sum(water['values'][-1]) > sum(water['values'][0])
    # The wet face carries a nonzero vapor rate; only the adiabatic
    # edges stay moisture-sealed (ticket 26: the cold-face convection
    # edge runs the same Robin pair and may exchange small amounts).
    final_rates = edge_water['values'][-1]
    assert abs(final_rates[0]) > 1e-9  # Hot Air, edge order begins there
    assert final_rates[4:] == pytest.approx([0.0] * 4, abs=1e-12)
    # The heat side is unaffected in shape: convection rows stay nonzero.
    assert any(abs(rate) > 1.0 for rate in edge_heat['values'][-1][:4])


def test_hygrothermal_moisture_budget_closes(tmp_path):
    """Boundary exchange accounts for the whole change in stored water.

    Ticket 26: Neumann convection edges always carry the moisture Robin
    pair, so the old "sealed domain" premise no longer exists. The
    honest invariant is conservation: the per-node water-content change
    (weighted by nodal volumes) cancels the kernel's accumulated outward
    WaterFluxStepExchange, per step.
    """
    def sealed_hook(members):
        bcs = ET.fromstring(members['SteadyStateBC.xml'])
        for record in bcs.findall('BoundaryCondition'):
            for tag in ('Simplified', 'Comprehensive'):
                block = record.find(tag)
                if block is not None:
                    humidity = block.find('RelativeHumidity')
                    if humidity is None:
                        ET.SubElement(block, 'RelativeHumidity').text = '0.5'
        members['SteadyStateBC.xml'] = ET.tostring(bcs)
        model = ET.fromstring(members['Model.xml'])
        for boundary in model.findall('Boundaries/Boundary'):
            transient_data = boundary.find('TransientData')
            if transient_data is not None:
                boundary.remove(transient_data)
        members['Model.xml'] = ET.tostring(model)

    source = _wet_slab(tmp_path, 'sealed.thmz', hot_transient=False,
                       type_records=[], timestep_files={},
                       members_hook=sealed_hook)
    result = solve(source)
    assert result.mode == 'hygrothermal-transient'
    water = _read(result.members, 'WaterContent.csv')
    exchange = _read(result.members, 'WaterFluxStepExchange.csv')
    exercised = False
    for index in range(1, len(exchange['times'])):
        outward = math.fsum(exchange['values'][index])
        delta = math.fsum(
            (a - b) * volume for a, b, volume in zip(
                water['values'][index], water['values'][index - 1],
                _NODAL_VOLUMES))
        exercised = exercised or abs(outward) > 0.0
        assert abs(outward + delta) <= 1e-2 * abs(outward) + 1e-9
    # The ledger was exercised by real Robin exchange, not an empty run.
    assert exercised


def test_hygrothermal_temperature_humidity_pin(tmp_path):
    """UseTemperature+UseHumidity pins both state variables (Dirichlet)."""
    record = _transient_type_bc(
        bc_model='Dirichlet', use_temperature=True, use_humidity=True)
    record['convection'] = None
    timestep = {HOT_FILE: save_timestep_file(TimestepFile(
        name='Hot Air transient',
        temperature=({'index': 1, 'temperature': 30.0},),
        humidity=({'index': 1, 'humidity': 0.7},)))}
    source = _wet_slab(tmp_path, type_records=[record],
                       timestep_files=timestep)
    result = solve(source)
    humidities = _read(result.members, 'Humidities.csv')
    temperatures = _read(result.members, 'Temperatures.csv')
    # The pinned boundary covers the three left-column nodes.
    left = (0, 3, 6)
    for row in (humidities['values'][-1], temperatures['values'][-1]):
        assert [row[i] for i in left] == pytest.approx(
            [0.7 if row is humidities['values'][-1] else 30.0] * 3,
            abs=1e-9)


def test_hygrothermal_humidity_only_pin(tmp_path):
    """A Dirichlet record with UseHumidity alone pins phi only."""
    record = _transient_type_bc(bc_model='Dirichlet', use_humidity=True)
    record['convection'] = None
    timestep = {HOT_FILE: save_timestep_file(TimestepFile(
        name='Hot Air transient',
        humidity=({'index': 1, 'humidity': 0.7},)))}
    source = _wet_slab(tmp_path, type_records=[record],
                       timestep_files=timestep)
    result = solve(source)
    humidities = _read(result.members, 'Humidities.csv')
    assert [humidities['values'][-1][i] for i in (0, 3, 6)] == \
        pytest.approx([0.7] * 3, abs=1e-9)


def test_hygrothermal_dirichlet_pin_wins_over_convection_record(tmp_path):
    """Dirichlet + UseHumidity pins phi even when a Convection sub-record
    exists: bcModel routes the BC, so the convection declaration stays inert
    metadata on a pinned edge (it must not become a humid Robin pair)."""
    record = _transient_type_bc(
        bc_model='Dirichlet', use_humidity=True,
        convection={'model': 'Fixed Convection Coefficient'})
    timestep = {HOT_FILE: save_timestep_file(TimestepFile(
        name='Hot Air transient',
        convection={'FixedConvectionFilmTimestep': _film_series(
            [20.0], humidity=0.9)},
        humidity=({'index': 1, 'humidity': 0.7},)))}
    source = _wet_slab(tmp_path, type_records=[record],
                       timestep_files=timestep)
    result = solve(source)
    humidities = _read(result.members, 'Humidities.csv')
    # A humidity pin holds exactly 0.7; a wet-Robin route would drift with
    # the convection series humidity (0.9) instead.
    assert [humidities['values'][-1][i] for i in (0, 3, 6)] == \
        pytest.approx([0.7] * 3, abs=1e-9)


def test_hygrothermal_dirichlet_pin_requires_humidity_series(tmp_path):
    """Dirichlet + UseHumidity with convection but no HumidityTimestep and
    no record Humidity must fail with a coded error, not route to Robin."""
    record = _transient_type_bc(
        bc_model='Dirichlet', use_humidity=True,
        convection={'model': 'Fixed Convection Coefficient'})
    timestep = {HOT_FILE: save_timestep_file(TimestepFile(
        name='Hot Air transient',
        convection={'FixedConvectionFilmTimestep': _film_series(
            [20.0], humidity=0.9)}))}
    source = _wet_slab(tmp_path, type_records=[record],
                       timestep_files=timestep)
    with pytest.raises(ContractError) as caught:
        solve(source)
    assert caught.value.code in ('THMZ-E4000', 'THMZ-E2003')


def test_hygrothermal_pin_needs_no_convection_series(tmp_path):
    """Pinned edges treat Convection as inert metadata: a transient record
    declaring the model needs no ``ConvectionTimesteps`` series at all
    (contract section 5). Both pin styles are covered."""
    for use_temperature, bc_model in ((False, 'Dirichlet'),
                                      (True, 'Dirichlet')):
        record = _transient_type_bc(
            bc_model=bc_model, use_temperature=use_temperature,
            use_humidity=True,
            convection={'model': 'Fixed Convection Coefficient'})
        file_args = {'name': 'Hot Air transient',
                     'humidity': ({'index': 1, 'humidity': 0.7},)}
        if use_temperature:
            file_args['temperature'] = (
                {'index': 1, 'temperature': 30.0},)
        timestep = {HOT_FILE: save_timestep_file(
            TimestepFile(**file_args))}
        source = _wet_slab(
            tmp_path, 'pin-t{}.thmz'.format(use_temperature),
            type_records=[record], timestep_files=timestep)
        result = solve(source)
        humidities = _read(result.members, 'Humidities.csv')
        assert [humidities['values'][-1][i] for i in (0, 3, 6)] == \
            pytest.approx([0.7] * 3, abs=1e-9)


def test_hygrothermal_pin_treats_radiation_and_flux_as_inert(tmp_path):
    """Radiation and UseHeatFlux declarations on a pinned edge are inert:
    no RadiationTimesteps/HeatFluxTimestep series are required and the
    declared Automatic Enclosure model is not rejected on the pin path."""
    record = _transient_type_bc(
        bc_model='Dirichlet', use_humidity=True,
        convection={'model': 'Fixed Convection Coefficient'},
        radiation={'model': 'Automatic Enclosure'},
        use_heat_flux=True, heat_flux=5.0)
    timestep = {HOT_FILE: save_timestep_file(TimestepFile(
        name='Hot Air transient',
        humidity=({'index': 1, 'humidity': 0.7},)))}
    source = _wet_slab(tmp_path, type_records=[record],
                       timestep_files=timestep)
    result = solve(source)
    humidities = _read(result.members, 'Humidities.csv')
    assert [humidities['values'][-1][i] for i in (0, 3, 6)] == \
        pytest.approx([0.7] * 3, abs=1e-9)


def test_hygrothermal_steady_record_humid_convection(tmp_path):
    """bcType=SteadyState humid convection takes the record Humidity."""
    record = _transient_type_bc(
        bc_type='SteadyState', use_humidity=True, humidity=0.9,
        convection={'model': 'Fixed Convection Coefficient',
                    'film_coefficient': 10.0, 'temperature': 20.0})
    timestep = {HOT_FILE: save_timestep_file(
        TimestepFile(name='Hot Air transient'))}
    source = _wet_slab(tmp_path, type_records=[record],
                       timestep_files=timestep)
    result = solve(source)
    assert result.mode == 'hygrothermal-transient'
    humidities = _read(result.members, 'Humidities.csv')
    assert max(humidities['values'][-1]) > INITIAL_HUMIDITY + 1e-4


def _freeze_hook(members):
    """Sub-zero cold side for the fusion budget.

    Ticket 26: a Neumann convection edge always carries the moisture
    Robin pair, so the old sealed-moisture premise is unreachable on a
    convective boundary. The cold face keeps its -50 C convection drive
    with its declared RelativeHumidity; the Robin water exchange there
    does not enter the heat ledger (no boundary enthalpy transport is
    modelled), so the budget check stays valid.
    """
    bcs = ET.fromstring(members['SteadyStateBC.xml'])
    for record in bcs.findall('BoundaryCondition'):
        name = record.findtext('Name')
        if name != 'Cold Air':
            continue
        block = record.find('Comprehensive')
        if block is None:
            block = record.find('Simplified')
        block.find('Convection/Temperature').text = '-50'
    members['SteadyStateBC.xml'] = ET.tostring(bcs)


# Kernel constants (vendor/HygroThermFEM Common.hxx): the liquid fraction is
# a linear ramp between the ice point and the freezing point.
_L_FUSION = 333550.0  # J/kg
_ICE_POINT = -0.1
_FREEZING_POINT = 0.0
# Consistent-mass nodal volumes of the 3x3-node slab mesh (1x0.5 m Q4s):
# corners get a quarter, edge nodes two quarters, the centre a full element.
_NODAL_VOLUMES = (0.125, 0.25, 0.125, 0.25, 0.5, 0.25, 0.125, 0.25, 0.125)
_CP_WATER = 4206.0
_CP_ICE = 2106.0
_RHO_ICE = 916.0
_RHO_WATER = 1000.0


def _liquid_fraction(temperature):
    if temperature <= _ICE_POINT:
        return 0.0
    if temperature >= _FREEZING_POINT:
        return 1.0
    return (temperature - _ICE_POINT) / (_FREEZING_POINT - _ICE_POINT)


def _nodal_energy(temperatures, water, fusion):
    """Sensible + latent internal energy per unit depth (J/m).

    Mirrors the kernel bookkeeping: with ``excludeLatentHeatOfFusion``
    disabled the element capacitance books
    ``(rho + liquid + ice) * (cp + ice*Cp_Ice/rho_ice +
    liquid*Cp_Water/rho_water)`` with the liquid fraction lambda(T)
    clamped to [IcePoint, FreezingPoint], plus ``L_f * lambda * w``
    latent; with the flag on, liquidPercent stays 1 (lambda = 1).
    """
    sensible = 0.0
    latent = 0.0
    for volume, temperature, w in zip(_NODAL_VOLUMES, temperatures, water):
        liquid = _liquid_fraction(temperature) * w if fusion else w
        ice = w - liquid
        capacity = (1.0 + w) * (1.0 + ice / _RHO_ICE * _CP_ICE
                                + liquid / _RHO_WATER * _CP_WATER)
        sensible += volume * capacity * temperature
        latent += volume * _L_FUSION * liquid
    return sensible, latent


def test_hygrothermal_latent_heat_budget(tmp_path):
    """Freezing closes the energy budget only when the latent term is kept.

    Sealed wet slab cooled below the ice point on its cold face while the
    hot face stays at 20 C: part of the stored water freezes. With
    ExcludeLatentHeatOfFusion=false the cumulative outward boundary heat
    must equal the sensible + latent internal-energy drop; repeating the
    same model with the flag excluded must leak the released latent heat.
    rho=cp=1 and dt=0.5 s match the dry reference case, so the same
    timestep is defensible. Internal evidence, not a THERM comparison.
    """
    results = {}
    for freeze in (True, False):
        source = _wet_slab(
            tmp_path, 'freeze-{}.thmz'.format(freeze),
            hot_transient=False, type_records=[], timestep_files={},
            bulk_density=1.0, heat_capacity=1.0, freeze=freeze,
            sorption_curve=((0.0, 0.0), (0.5, 10.0), (1.0, 20.0)),
            members_hook=_freeze_hook)
        results[freeze] = solve(source)
    for freeze, result in results.items():
        assert result.mode == 'hygrothermal-transient'
        # Use the kernel's accumulated interval exchange: instant-rate
        # HeatFluxEdges cannot be trusted on a UseTemperature pin edge
        # (huge-film reintegration noise dwarfs the physical rate).
        edges = _read(result.members, 'HeatFluxStepExchange.csv')
        temperatures = _read(result.members, 'Temperatures.csv')
        water = _read(result.members, 'WaterContent.csv')
        net_outward = math.fsum(
            math.fsum(row) for row in edges['values'][1:])
        e0 = _nodal_energy(temperatures['values'][0],
                           water['values'][0], freeze)
        e1 = _nodal_energy(temperatures['values'][-1],
                           water['values'][-1], freeze)
        delta_sensible = e1[0] - e0[0]
        delta_latent = e1[1] - e0[1]
        # Residuals of net_outward + dE_sensible (+ dE_latent) = 0. The
        # no-latent residual is the bookkeeping each run must satisfy:
        # fusion-on books the released latent heat, fusion-off does not.
        with_latent = abs(net_outward + delta_sensible + delta_latent)
        without_latent = abs(net_outward + delta_sensible)
        scale = max(abs(delta_latent), abs(delta_sensible), 1.0)
        results[freeze] = dict(with_latent=with_latent,
                               without_latent=without_latent,
                               scale=scale, latent=delta_latent,
                               cold=temperatures['values'][-1][2])
    frozen = results[True]
    melted = results[False]
    # Freezing actually happened: the cold-side nodes crossed the ramp and
    # the run books released latent heat (dE_latent < 0).
    assert frozen['cold'] < _FREEZING_POINT
    assert frozen['latent'] < -1e3
    # With the term enabled the latent-inclusive budget closes within 5%
    # of its scale while the same bookkeeping without the latent term
    # visibly fails (it misses the released heat); the fusion-off run
    # instead closes exactly on its own sensible-only bookkeeping.
    assert frozen['with_latent'] <= 0.05 * frozen['scale']
    assert frozen['without_latent'] > 0.5 * abs(frozen['latent'])
    assert melted['without_latent'] <= 0.05 * melted['scale']


def test_hygrothermal_timestep_halving_converges(tmp_path):
    """Wet timestep convergence: halving dt moves fields by < 5e-3.

    Internal evidence on the humid-convection case (not a THERM
    comparison): the implicit first-order scheme must stay inside a fixed
    band under refinement, mirroring the dry halving check of ticket 06.
    """
    record, timestep = _humid_convection(humidity=0.9)
    coarse = _wet_slab(tmp_path, 'wet-coarse.thmz',
                       type_records=[record], timestep_files=timestep)
    fine = _wet_slab(tmp_path, 'wet-fine.thmz',
                     type_records=[record], timestep_files=timestep,
                     time_step=TIME_STEP / 2, num_steps=NUM_STEPS * 2)
    coarse_result = solve(coarse)
    fine_result = solve(fine)
    for member in ('Temperatures.csv', 'Humidities.csv',
                   'WaterContent.csv'):
        coarse_rows = _read(coarse_result.members, member)['values'][-1]
        fine_rows = _read(fine_result.members, member)['values'][-1]
        unit = 10.0 if member == 'WaterContent.csv' else 1.0
        diff = max(abs(a - b) for a, b in zip(coarse_rows, fine_rows))
        assert diff < 5e-3 * unit, member
        assert diff > 0.0, member


def test_hygrothermal_mesh_refinement_converges():
    """Wet mesh convergence: net water uptake shifts < 5% under halving.

    Direct-binding comparison of the 4-quad slab with a 16-quad twin (same
    2x1 m face, same humid Robin drive on the left face). The nodal
    capacity weighting uses per-element quarter volumes so both meshes
    integrate the same domain. Internal evidence, not a THERM comparison.
    """
    from pyhygrothermfem_fairyfly import _native

    def humid_edges(nx, ny):
        """Left face edges: node pairs of the first column, per row."""
        return [(row * (nx + 1) + 1, (row + 1) * (nx + 1) + 1)
                for row in range(ny)]

    def grid(nx, ny):
        nodes = []
        quads = []
        for row in range(ny + 1):
            for col in range(nx + 1):
                nodes.append((len(nodes) + 1, col * 2.0 / nx,
                              row * 1.0 / ny))
        for row in range(ny):
            for col in range(nx):
                a = row * (nx + 1) + col + 1
                b = a + 1
                d = a + nx + 1
                c = d + 1
                quads.append((a, b, c, d))
        humid = [(a, b, (10.0,), (20.0,), (0.9,))
                 for a, b in humid_edges(nx, ny)]
        # Right face: dry convection at 0 C (film 10, no moisture exchange).
        right = [(row * (nx + 1) + nx + 1, (row + 1) * (nx + 1) + nx + 1)
                 for row in range(ny)]
        dry = [(a, b, (10.0,), (0.0,)) for a, b in right]
        materials = [(1.0, 0.9, 0.9, '', '', '', False, 1.0, 101325.0)] \
            * len(quads)
        hygro = [(600.0, 0.2, 850.0, 50.0, list(SORPTION), [], [], 0.0,
                  [], 0.0)] * len(quads)
        physics = (True, True, True, True, True, False)
        return nodes, quads, materials, hygro, physics, humid, dry

    def total_water(nodes, quads, water):
        volumes = [0.0] * len(nodes)
        positions = {nid: (x, y) for nid, x, y in nodes}
        for a, b, c, d in quads:
            ax, ay = positions[a]
            bx, by = positions[b]
            cx, cy = positions[c]
            dx, dy = positions[d]
            area = 0.5 * abs((bx - ax) * (cy - ay)
                             - (cx - ax) * (by - ay)) \
                + 0.5 * abs((cx - ax) * (dy - ay)
                            - (dx - ax) * (cy - ay))
            for corner in (a, b, c, d):
                volumes[corner - 1] += area / 4.0
        return sum(volume * w for volume, w in zip(volumes, water))

    uptakes = []
    for nx in (2, 4):
        nodes, quads, materials, hygro, physics, humid, dry = grid(nx, 2)
        frames = []
        (_t, _f, _h, _wf, steps, _slots) = _native.solve_hygrothermal(
            nodes, quads, materials, hygro, physics,
            convection=dry, humid_convection=humid,
            initial_temperature=10.0, initial_humidity=0.5,
            time_step=TIME_STEP, num_steps=NUM_STEPS,
            convergence_tolerance=1e-8, relaxation=1.0,
            on_step=frames.append)
        assert steps == NUM_STEPS
        initial = total_water(nodes, quads, frames[0][4])
        final = total_water(nodes, quads, frames[-1][4])
        uptakes.append(final - initial)
    assert all(uptake > 0.0 for uptake in uptakes)
    # Preset bound: halving the element size moves the net uptake by less
    # than 5% of its magnitude.
    assert abs(uptakes[1] - uptakes[0]) <= 0.05 * abs(uptakes[0])


def test_hygrothermal_periodic_bc_types_match_expanded_series():
    """Wet periodic BCs reproduce the lcm-expanded-table semantics.

    Per-field cyclic indexing is a memory optimization: humid convection
    (film/temperature/humidity), the humidity pin and the combined
    temperature+humidity pin must produce identical node series as their
    lcm-expanded equivalents.
    """
    from pyhygrothermfem_fairyfly import _native
    nodes = [(index + 1, (index % 3) * 1.0, (index // 3) * 0.5)
             for index in range(9)]
    quads = [(1, 2, 5, 4), (2, 3, 6, 5), (4, 5, 8, 7), (5, 6, 9, 8)]
    materials = [(1.0, 0.9, 0.9, '', '', '', False, 1.0, 101325.0)] * 4
    hygro = [(600.0, 0.2, 850.0, 50.0, list(SORPTION), [], [], 0.0,
              [], 0.0)] * 4
    physics = (True, True, True, True, True, False)

    def run(**kwargs):
        frames = []
        out = _native.solve_hygrothermal(
            nodes, quads, materials, hygro, physics,
            initial_temperature=10.0, initial_humidity=0.5,
            time_step=TIME_STEP, num_steps=14,
            convergence_tolerance=1e-8, relaxation=1.0,
            on_step=frames.append, **kwargs)
        assert out[4] == 14
        return frames, out[5]

    anchor = [(2, 3, (10.0,), (0.0,))]

    # Humid convection: film period 2, temperature period 3, humidity
    # period 2 -- combined lcm 6.
    frames, slots = run(
        convection=anchor,
        humid_convection=[
            (1, 2, (10.0, 12.0), (20.0, 5.0, 15.0), (0.9, 0.5))])
    assert slots == 7
    expanded, _slots = run(
        convection=anchor,
        humid_convection=[
            (1, 2, (10.0, 12.0) * 3, (20.0, 5.0, 15.0) * 2, (0.9, 0.5) * 3)])
    assert [f[3] for f in frames] == [f[3] for f in expanded]
    assert [f[4] for f in frames] == [f[4] for f in expanded]

    # Humidity-only pin: phi period 3.
    frames, slots = run(
        convection=anchor,
        humidities=[(1, 2, (0.9, 0.5, 0.7))])
    assert slots == 3
    expanded, _slots = run(
        convection=anchor,
        humidities=[(1, 2, (0.9, 0.5, 0.7, 0.9, 0.5, 0.7))])
    assert [f[3] for f in frames] == [f[3] for f in expanded]

    # Temperature+humidity pin: T period 2, phi period 3 (lcm 6).
    frames, slots = run(
        convection=anchor,
        temperature_humidities=[
            (1, 2, (30.0, 10.0), (0.9, 0.5, 0.7))])
    assert slots == 5
    expanded, _slots = run(
        convection=anchor,
        temperature_humidities=[
            (1, 2, (30.0, 10.0) * 3, (0.9, 0.5, 0.7) * 2)])
    assert [f[2] for f in frames] == [f[2] for f in expanded]
    assert [f[3] for f in frames] == [f[3] for f in expanded]


def test_hygrothermal_periodic_pins_seed_t0():
    """Periodic pins seed the t=0 frame like their fixed counterparts.

    The temperature side shares PeriodicTemperatureBC with the dry path
    (moved-from series read seeded 0 C); the humidity side never seeds
    node phi (same as MoistureBCFixedHumidity), so phi stays at the
    declared initial until the first solve applies the series. Fixed
    (single-entry) pins are the unchanged reference forms.
    """
    from pyhygrothermfem_fairyfly import _native
    nodes = [(index + 1, (index % 3) * 1.0, (index // 3) * 0.5)
             for index in range(9)]
    quads = [(1, 2, 5, 4), (2, 3, 6, 5), (4, 5, 8, 7), (5, 6, 9, 8)]
    materials = [(1.0, 0.9, 0.9, '', '', '', False, 1.0, 101325.0)] * 4
    hygro = [(600.0, 0.2, 850.0, 50.0, list(SORPTION), [], [], 0.0,
              [], 0.0)] * 4
    physics = (True, True, True, True, True, False)
    anchor = [(2, 3, (10.0,), (0.0,))]

    def run(**kwargs):
        frames = []
        out = _native.solve_hygrothermal(
            nodes, quads, materials, hygro, physics,
            convection=anchor,
            initial_temperature=10.0, initial_humidity=0.5,
            time_step=0.5, num_steps=3,
            convergence_tolerance=1e-8, relaxation=1.0,
            on_step=frames.append, **kwargs)
        assert out[4] == 3
        return frames

    # Combined temperature+humidity pin: T seeds from series[0].
    periodic = run(temperature_humidities=[
        (1, 2, (30.0, 5.0), (0.9, 0.5))])
    fixed = run(temperature_humidities=[(1, 2, (30.0,), (0.9,))])
    assert periodic[0][2][:2] == pytest.approx([30.0, 30.0], abs=1e-9)
    assert periodic[0][2][:2] == fixed[0][2][:2]
    assert periodic[0][3][:2] == pytest.approx([0.5, 0.5], abs=1e-9)
    assert periodic[0][3][:2] == fixed[0][3][:2]
    # Step 0 applies the first series entry on both state variables.
    assert periodic[1][2][:2] == pytest.approx([30.0, 30.0], abs=1e-9)
    assert periodic[1][3][:2] == pytest.approx([0.9, 0.9], abs=1e-9)

    # Humidity-only pin: no seeding on either variable at t=0; the pin
    # applies series[0] = 0.9 from step 0 (fixed pin behaves the same).
    periodic = run(humidities=[(1, 2, (0.9, 0.5))])
    fixed = run(humidities=[(1, 2, (0.9,))])
    assert periodic[0][3][:2] == pytest.approx([0.5, 0.5], abs=1e-9)
    assert periodic[0][3][:2] == fixed[0][3][:2]
    assert periodic[1][3][:2] == pytest.approx([0.9, 0.9], abs=1e-9)


def test_hygrothermal_run_thmz_publishes_transient_members(tmp_path):
    """run_thmz streams all seven transient members for wet solves."""
    record, timestep = _humid_convection()
    source = _wet_slab(tmp_path, type_records=[record],
                       timestep_files=timestep)
    target = tmp_path / 'wet_solved.thmz'
    published = Path(run_thmz(source, output=target))
    with zipfile.ZipFile(published) as archive:
        names = set(archive.namelist())
        assert set(series_io.TRANSIENT_RESULT_MEMBERS.values()) <= names
        assert 'SteadyStateResults.xml' not in names
        assert 'SteadyStateMeshResults.xml' not in names
        humidities = series_io.read_transient_series(
            archive.read('transient results/Humidities.csv'))
        assert len(humidities['values']) == NUM_STEPS + 1


# --- explicit rejections ---------------------------------------------------

def test_hygrothermal_rejects_steady_moisture(tmp_path):
    """SimulateMoisture without Transient CalculationMode stays E5003."""
    members = _slab_members()
    model = ET.fromstring(members['Model.xml'])
    model.find(
        'Properties/CalculationOptions/SimulateMoisture').text = 'true'
    members['Model.xml'] = ET.tostring(model)
    target = tmp_path / 'steady_wet.thmz'
    with zipfile.ZipFile(target, 'w') as archive:
        for name, content in members.items():
            archive.writestr(name, content)
    with pytest.raises(UnsupportedError) as caught:
        solve(target)
    assert caught.value.code == 'THMZ-E5003'


def test_hygrothermal_pin_still_rejects_montazeri(tmp_path):
    """A humidity-pinned transient record keeps the model-level E5001.

    The pin makes the declared Convection inert metadata (no series are
    required), but Montazeri is a model-level unsupported item: it must
    be rejected, not silently ignored via the pin.
    """
    record = _transient_type_bc(
        bc_model='Dirichlet', use_humidity=True,
        convection={'model': 'Montazeri'})
    timestep = {HOT_FILE: save_timestep_file(TimestepFile(
        name='Hot Air transient',
        humidity=({'index': 1, 'humidity': 0.7},)))}
    source = _wet_slab(tmp_path, type_records=[record],
                       timestep_files=timestep)
    with pytest.raises(UnsupportedError) as caught:
        solve(source)
    assert caught.value.code == 'THMZ-E5001'


def test_hygrothermal_steady_record_rejects_montazeri(tmp_path):
    """bcType=SteadyState humidity-pinned records keep E5001 as well.

    Before the fix the SteadyType early exit skipped every per-record
    check, so a pinned record declaring Montazeri would solve silently
    (the pin treats convection as inert) -- the most dangerous skip.
    """
    record = _transient_type_bc(
        bc_type='SteadyState', bc_model='Dirichlet', use_humidity=True,
        humidity=0.9, convection={'model': 'Montazeri'})
    timestep = {HOT_FILE: save_timestep_file(
        TimestepFile(name='Hot Air transient'))}
    source = _wet_slab(tmp_path, type_records=[record],
                       timestep_files=timestep)
    with pytest.raises(UnsupportedError) as caught:
        solve(source)
    assert caught.value.code == 'THMZ-E5001'


def test_hygrothermal_rejects_neumann_humidity_without_convection(tmp_path):
    """UseHumidity on a Neumann record without convection is E5016."""
    record = _transient_type_bc(bc_model='Neumann', use_humidity=True)
    record['convection'] = None
    timestep = {HOT_FILE: save_timestep_file(TimestepFile(
        name='Hot Air transient',
        humidity=({'index': 1, 'humidity': 0.5},)))}
    source = _wet_slab(tmp_path, type_records=[record],
                       timestep_files=timestep)
    with pytest.raises(UnsupportedError) as caught:
        solve(source)
    assert caught.value.code == 'THMZ-E5016'


def test_hygrothermal_rejects_mixed_conductivity_options(tmp_path):
    """The two conductivity excludes must agree (E5010)."""
    def hook(members):
        model = ET.fromstring(members['Model.xml'])
        model.find('Properties/CalculationOptions/ModelingOptions/'
                   'ExcludeThermalConductivityTemperatureDependent'
                   ).text = 'false'
        members['Model.xml'] = ET.tostring(model)

    source = _wet_slab(tmp_path, members_hook=hook)
    with pytest.raises(UnsupportedError) as caught:
        solve(source)
    assert caught.value.code == 'THMZ-E5010'


def test_hygrothermal_rejects_unsupported_material_tables(tmp_path):
    """Redistribution, mu(w) and enthalpy tables are declared unsupported."""
    for tag in ('LiquidTransportationCoefficientRedistribution',
                'WaterVaporDiffusionResistanceFactorMoistureDependent',
                'EnthalpyTemperatureDependent'):
        def hook(members, tag=tag):
            mats = ET.fromstring(members['Materials.xml'])
            hygro = mats.find('Material/Solid/HygroThermal')
            table = ET.SubElement(hygro, tag)
            point = ET.SubElement(table, 'TableValue')
            ET.SubElement(point, 'x').text = '0'
            ET.SubElement(point, 'y').text = '1'
            members['Materials.xml'] = ET.tostring(mats)

        source = _wet_slab(tmp_path, 'wet-{}.thmz'.format(tag),
                           members_hook=hook)
        with pytest.raises(UnsupportedError) as caught:
            solve(source)
        assert caught.value.code == 'THMZ-E5010', tag


def test_hygrothermal_requires_moisture_fields(tmp_path):
    """Wet materials need porosity and the sorption curve (E2003)."""
    def no_porosity(members):
        mats = ET.fromstring(members['Materials.xml'])
        hygro = mats.find('Material/Solid/HygroThermal')
        ET.SubElement(hygro, 'BulkDensity').text = '600'
        ET.SubElement(hygro, 'SpecificHeatCapacityDry').text = '850'
        ET.SubElement(
            hygro, 'WaterVaporDiffusionResistanceFactor').text = '50'
        storage = ET.SubElement(hygro, 'MoistureStorageFunction')
        for x, y in SORPTION:
            point = ET.SubElement(storage, 'TableValue')
            ET.SubElement(point, 'x').text = repr(x)
            ET.SubElement(point, 'y').text = repr(y)
        members['Materials.xml'] = ET.tostring(mats)
        model = ET.fromstring(members['Model.xml'])
        model.find(
            'Properties/CalculationOptions/SimulateMoisture').text = 'true'
        members['Model.xml'] = ET.tostring(model)

    source = _transient_slab(tmp_path, 'wet.thmz', density=False,
                             members_hook=no_porosity)
    with pytest.raises(ContractError) as caught:
        solve(source)
    assert caught.value.code == 'THMZ-E2003'


def test_hygrothermal_requires_liquid_table_when_enabled(tmp_path):
    """Liquid transport enabled without the suction table is E2004."""
    def hook(members):
        model = ET.fromstring(members['Model.xml'])
        model.find('Properties/CalculationOptions/ModelingOptions/'
                   'ExcludeWaterLiquidTransportation').text = 'false'
        members['Model.xml'] = ET.tostring(model)

    source = _wet_slab(tmp_path, 'liquid.thmz', members_hook=hook)
    with pytest.raises(ContractError) as caught:
        solve(source)
    assert caught.value.code == 'THMZ-E2004'


def test_hygrothermal_kernel_failure_is_solve_error(tmp_path, monkeypatch):
    """A wet kernel failure surfaces as SolveError, never a partial result."""
    from pyhygrothermfem_fairyfly import _native

    def fail_solve(*args, **kwargs):
        raise RuntimeError('hygrothermal kernel did not converge')

    monkeypatch.setattr(_native, 'solve_hygrothermal', fail_solve)
    record, timestep = _humid_convection()
    source = _wet_slab(tmp_path, type_records=[record],
                       timestep_files=timestep)
    with pytest.raises(SolveError) as caught:
        solve(source)
    assert caught.value.code is None
