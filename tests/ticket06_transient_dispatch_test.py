"""Ticket 06: transient dispatch, streaming results, coded rejections.

``slab.thmz`` is patched into a transient twin that reproduces the
``native_reference`` C++ benchmark case (same mesh, convection boundaries,
dt=0.5 s, 400 steps, unit rho*cp, initial 10 C) so the end-to-end Python
path is checked field for field against the kernel reference. The transient
members (TransientTypeBC.xml + timestep files) are schema-mirrored; layouts
not yet seen in a real THERM 8.1 export stay ``unverified`` per contract
section 9 (ticket 07).
"""
import csv
import io
import json
from pathlib import Path
import zipfile
import xml.etree.ElementTree as ET

import pytest
from ticket18_boundary_fixtures import declare_constant_boundaries

from pyhygrothermfem_fairyfly import (ContractError, SolveError, UnsupportedError,
                          inspect as inspect_thmz, run_thmz, solve)
from pyhygrothermfem_fairyfly import transient_results as series_io
from pyhygrothermfem_fairyfly.transient_input import (
    TIMESTEP_DIR, TimestepFile, save_timestep_file, save_transient_type_bc)

SLAB = Path(__file__).parent / 'assets' / 'slab.thmz'
REFERENCE = json.loads(
    (Path(__file__).parent / 'assets' / 'native_reference.json').read_text())
TRANSIENT_REF = REFERENCE['transient']
NUM_STEPS = TRANSIENT_REF['num_steps']
TIME_STEP = TRANSIENT_REF['time_step']

HOT_UUID = 'bc-hot-transient'
HOT_FILE = 'hot.xml'


def _slab_members():
    with zipfile.ZipFile(SLAB) as archive:
        return {name: archive.read(name) for name in archive.namelist()}


def _film_series(values, humidity=0.5):
    """FixedConvectionFilmTimestep entries with contiguous 1-based indices."""
    return tuple(
        {'index': step + 1, 'film_coefficient': 10.0, 'temperature': value,
         'humidity': humidity}
        for step, value in enumerate(values))


def _transient_type_bc(convection=None, **overrides):
    record = {
        'uuid': HOT_UUID, 'name': 'Hot Air transient', 'protected': False,
        'bc_type': 'Transient', 'bc_model': 'Neumann',
        'convection': {'model': 'Fixed Convection Coefficient'}
        if convection is None else convection,
        'radiation': None, 'solar': None,
        'use_heat_flux': False, 'heat_flux': None,
        'use_temperature': False, 'temperature': None,
        'use_humidity': False, 'humidity': None,
        'color': '0x08c4cd'}
    record.update(overrides)
    return record


def _transient_slab(tmp_path, name='transient.thmz', *, density=True,
                    methodology='Constant everywhere',
                    type_records=None, timestep_files=None,
                    hot_transient=True, members_hook=None,
                    time_step=None, num_steps=None):
    """Patch slab.thmz into the transient reference case.

    The Hot Air boundary carries a TransientData record whose timestep file
    expands cyclically to ``num_steps`` constant (h=10, T=20) coefficients —
    identical physics to the SteadyStateBC record it replaces. Cold Air and
    Cold Air uses explicit constant TransientData (ticket 18 decision A);
    both Adiabatic boundaries remain sealed, zero-exchange exceptions.
    The required transient type member is also emitted for an empty table.
    ``members_hook`` may further adjust the member map before writing.
    """
    members = _slab_members()

    model = ET.fromstring(members['Model.xml'])
    options = model.find('Properties/CalculationOptions')
    options.find('CalculationMode').text = 'Transient'
    options.find('TransientCalculationMethodology').text = methodology
    options.find('ConstantInitialConditionsTransient/Temperature').text = '10'
    engine = options.find('EngineParameters')
    engine.find('ConvergenceTolerance').text = '1e-10'
    engine.find('RelaxationParameter').text = '1'
    engine.find('TimeStep').text = repr(
        TIME_STEP if time_step is None else time_step)
    engine.find('NumberOfTimesteps').text = str(
        NUM_STEPS if num_steps is None else num_steps)
    if hot_transient:
        boundary = next(boundary for boundary in
                        model.findall('Boundaries/Boundary')
                        if boundary.findtext('ID') == '2')
        transient_data = ET.SubElement(boundary, 'TransientData')
        ET.SubElement(transient_data, 'UUID').text = HOT_UUID
        ET.SubElement(transient_data, 'TimestepFileName').text = HOT_FILE
    members['Model.xml'] = ET.tostring(model)

    materials = ET.fromstring(members['Materials.xml'])
    hygro = materials.find('Material/Solid/HygroThermal')
    if density:
        ET.SubElement(hygro, 'BulkDensity').text = '1.0'
        ET.SubElement(hygro, 'SpecificHeatCapacityDry').text = '1.0'
    members['Materials.xml'] = ET.tostring(materials)

    if type_records is None:
        type_records = [_transient_type_bc()] if hot_transient else []
    members['TransientTypeBC.xml'] = save_transient_type_bc(
        type_records).encode('utf-8')
    if timestep_files is None and hot_transient:
        timestep_files = {HOT_FILE: save_timestep_file(TimestepFile(
            name='Hot Air transient',
            convection={'FixedConvectionFilmTimestep': _film_series(
                [20.0])}))}
    for file_name, content in (timestep_files or {}).items():
        members[TIMESTEP_DIR + '/' + file_name] = (
            content.encode('utf-8') if isinstance(content, str) else content)

    if members_hook is not None:
        members_hook(members)

    declare_constant_boundaries(members)

    target = tmp_path / name
    with zipfile.ZipFile(target, 'w') as archive:
        for member, content in members.items():
            archive.writestr(member, content)
    return target


def _read_csv(content):
    return series_io.read_transient_series(content)


# --- end-to-end dispatch ----------------------------------------------------

def test_inspect_reports_transient_mode(tmp_path):
    report = inspect_thmz(_transient_slab(tmp_path))
    assert report.mode == 'Transient'
    assert report.ok
    assert 'TransientTypeBC.xml' in report.members
    assert 'timestep input files/hot.xml' in report.members


def test_transient_solve_matches_native_reference(tmp_path):
    """The binding reproduces native_reference.cxx field for field."""
    result = solve(_transient_slab(tmp_path))
    assert result.mode == 'dry-transient'
    assert result.steps_completed == NUM_STEPS
    assert result.energy_error_percent is None
    assert result.provenance['contract_version'] == 2001
    assert result.provenance['mode'] == 'dry-transient'
    assert result.provenance['time_step'] == TIME_STEP
    assert result.provenance['number_of_timesteps'] == NUM_STEPS

    # Final field equals the last C++ reference step bit for bit (CSV writes
    # repr(), which round-trips doubles exactly).
    members = result.members
    temperatures = _read_csv(
        members['transient results/Temperatures.csv'])
    assert temperatures['entities'] == tuple(range(1, 10))
    assert len(temperatures['values']) == NUM_STEPS + 1  # t=0 plus steps
    assert temperatures['times'][0] == 0.0
    assert temperatures['times'][1:] == pytest.approx(
        [TIME_STEP * (i + 1) for i in range(NUM_STEPS)])
    assert temperatures['values'][0] == pytest.approx([10.0] * 9)
    for row, expected in zip(temperatures['values'][1:],
                             TRANSIENT_REF['steps']):
        assert row == pytest.approx(expected, rel=1e-12, abs=1e-12)
    assert list(result.temperatures) == pytest.approx(
        TRANSIENT_REF['steps'][-1], rel=1e-12, abs=1e-12)
    flat_fluxes = [component for flux in result.heat_fluxes
                   for component in flux]
    flat_reference = [component for flux in TRANSIENT_REF['final_heat_flux']
                      for component in flux]
    assert flat_fluxes == pytest.approx(
        flat_reference, rel=1e-12, abs=1e-12)

    # After ~50 characteristic times the field sits on the steady solution.
    steady = [19.090909090909093, 10.0, 0.9090909090909091] * 3
    assert list(result.temperatures) == pytest.approx(steady, abs=1e-4)


def test_transient_solve_streams_all_result_members(tmp_path):
    """All seven CSV members stream one row per step plus the t=0 frame."""
    result = solve(_transient_slab(tmp_path))
    members = result.members
    expected = set(series_io.TRANSIENT_RESULT_MEMBERS.values())
    assert expected <= set(members)
    # Steady result members are not produced in transient mode.
    assert 'SteadyStateMeshResults.xml' not in members
    assert 'SteadyStateResults.xml' not in members
    # Input members pass through untouched, including transient inputs.
    for name, content in _slab_members().items():
        if name in ('Model.xml', 'Materials.xml', 'SteadyStateBC.xml'):
            continue  # patched for the transient twin
        assert members[name] == content
    assert members['TransientTypeBC.xml']
    assert members['timestep input files/hot.xml']

    geometry = series_io.read_geometry_xml(
        members['transient results/Geometry.xml'])
    assert len(geometry['nodes']) == 9
    assert len(geometry['elements']) == 4
    assert len(geometry['materials']) == 1
    assert len(geometry['model_enclosure']) == 8

    node_members = ('Temperatures.csv', 'Humidities.csv', 'HeatFlux.csv',
                    'WaterContent.csv', 'WaterFlux.csv')
    for short in node_members:
        series = _read_csv(
            members['transient results/' + short])
        assert series['entities'] == tuple(range(1, 10))
        assert len(series['values']) == NUM_STEPS + 1
    for short in ('HeatFluxEdges.csv', 'WaterFluxEdges.csv'):
        series = _read_csv(
            members['transient results/' + short])
        assert len(series['entities']) == 8
        assert len(series['values']) == NUM_STEPS + 1

    # Dry solve: humidity stays at the declared initial 0.5, water flux is
    # identically zero (never fabricated moisture physics).
    humidities = _read_csv(members['transient results/Humidities.csv'])
    assert all(value == pytest.approx(0.5)
               for row in humidities['values'] for value in row)

    # Edge order: boundary IDs 2,3,4,5 each ordered along the boundary line.
    # Final step sits on the steady profile: hot edges shed 4.5455 W/m
    # inward-negative, cold edges outward-positive, adiabatic edges zero.
    edges = _read_csv(members['transient results/HeatFluxEdges.csv'])
    final = edges['values'][-1]
    assert final[:4] == pytest.approx(
        [-20 / 4.4, -20 / 4.4, 20 / 4.4, 20 / 4.4], abs=1e-3)
    assert final[4:] == pytest.approx([0.0] * 4, abs=1e-9)


def test_run_thmz_publishes_transient_archive(tmp_path):
    source = _transient_slab(tmp_path)
    target = tmp_path / 'transient_solved.thmz'
    published = Path(run_thmz(source, output=target))
    expected_members = solve(source).members
    with zipfile.ZipFile(published) as archive:
        assert archive.namelist() == list(expected_members)
        for name in archive.namelist():
            assert archive.read(name) == expected_members[name]


def test_transient_series_stay_compact_to_native(monkeypatch, tmp_path):
    """Constant and periodic boundary series cross the binding unexpanded.

    Python must not materialize one list entry per timestep (ticket 06
    bounded-memory requirement): constant BCs pass a single value and a
    periodic series passes its raw entries; cyclic indexing happens in the
    kernel binding.
    """
    from pyhygrothermfem_fairyfly import _native
    seen = {}

    def spy(nodes, quads, materials, densities, heat_capacities,
            **kwargs):
        seen['convection'] = kwargs['convection']
        seen['fluxes'] = kwargs['fluxes']
        seen['temperatures'] = kwargs['temperatures']
        seen['radiations'] = kwargs['radiations']
        seen['linearized_radiations'] = kwargs['linearized_radiations']
        return _real(nodes, quads, materials, densities, heat_capacities,
                     **kwargs)

    _real = _native.solve_transient
    monkeypatch.setattr(_native, 'solve_transient', spy)
    timestep = {HOT_FILE: save_timestep_file(TimestepFile(
        name='Hot Air transient',
        convection={'FixedConvectionFilmTimestep': _film_series(
            [20.0, 5.0])}))}
    solve(_transient_slab(tmp_path, name='compact.thmz',
                          timestep_files=timestep))
    # The periodic two-entry driving series arrives as two entries, not
    # num_steps copies; explicit constant TransientTypeBC edges arrive as
    # one-element series.
    for edge in seen['convection']:
        assert len(edge[2]) <= 2 and len(edge[3]) <= 2
    for series in seen['fluxes'] + seen['temperatures'] + \
            seen['radiations'] + seen['linearized_radiations']:
        assert len(series[2]) <= 2


def test_transient_bc_tables_bounded_by_period(tmp_path, monkeypatch):
    """Kernel BC series slots allocate the field periods, not num_steps.

    ``solve_transient`` reports the total cyclic-series slots its periodic
    BCs hold (``bc_timestep_slots``). A two-entry periodic driving series
    must hold only its per-field periods regardless of
    ``NumberOfTimesteps``; sizing tables by num_steps (or combining fields
    into an lcm-period table) would regress the bounded-memory requirement
    on long runs (contract section 5).
    """
    from pyhygrothermfem_fairyfly import _native
    seen = []
    _real = _native.solve_transient

    def spy(nodes, quads, materials, densities, heat_capacities, **kwargs):
        result = _real(nodes, quads, materials, densities, heat_capacities,
                       **kwargs)
        seen.append(result[3])
        return result

    monkeypatch.setattr(_native, 'solve_transient', spy)
    timestep = {HOT_FILE: save_timestep_file(TimestepFile(
        name='Hot Air transient',
        convection={'FixedConvectionFilmTimestep': _film_series(
            [20.0, 5.0])}))}
    solve(_transient_slab(tmp_path, timestep_files=timestep))
    # Hot Air covers two mesh edges, each holding a two-entry film series
    # plus a two-entry temperature series (per-field indexing); the
    # constant fallback edges use scalar BCs and allocate no series slots.
    assert seen == [8]
    assert seen[0] < NUM_STEPS


def test_transient_mixed_period_series_use_lcm():
    """Differing field periods index cyclically, without lcm allocation.

    Direct binding check: a two-entry film series combined with a
    three-entry temperature series holds 2 + 3 = 5 cyclic slots. The
    expanded six-entry equivalent (the old lcm table content) must produce
    bitwise-identical node series, proving the per-field indexing keeps
    the same physics.
    """
    from pyhygrothermfem_fairyfly import _native
    nodes = [(index + 1, (index % 3) * 1.0, (index // 3) * 0.5)
             for index in range(9)]
    quads = [(1, 2, 5, 4), (2, 3, 6, 5), (4, 5, 8, 7), (5, 6, 9, 8)]
    materials = [(1.0, 0.9, 0.9, '', '', '', False, 1.0, 101325.0)] * 4
    edges = [(3, 6, (10.0,), (0.0,)), (6, 9, (10.0,), (0.0,))]

    def run(film, temperatures):
        frames = []
        _t, _f, steps, slots = _native.solve_transient(
            nodes, quads, materials, [1.0] * 4, [1.0] * 4,
            convection=[(1, 2, film, temperatures)] + edges,
            initial_temperature=10.0, initial_humidity=0.5,
            time_step=0.5, num_steps=13,
            convergence_tolerance=1e-8, relaxation=1.0,
            on_step=frames.append)
        return steps, slots, frames

    steps, slots, compact = run((10.0, 10.0), (20.0, 5.0, 15.0))
    assert steps == 13
    assert slots == 5
    # lcm(2, 3) = 6: expand each field to the lcm period. The expanded
    # series describe identical per-step coefficients.
    _steps, _slots, expanded = run(
        (10.0,) * 6, (20.0, 5.0, 15.0, 20.0, 5.0, 15.0))
    assert [frame[2] for frame in compact] == \
        [frame[2] for frame in expanded]


def test_transient_coprime_periods_stay_compact():
    """Coprime field periods never materialize the lcm table.

    Reproduction of the unbounded-allocation defect: film period 1009 and
    temperature period 1013 are coprime, so a combined lcm table would
    allocate 1,022,117 coefficient entries per edge -- far beyond any sane
    ``num_steps``. Per-field indexing holds 1,009 + 1,013 = 2,022 slots.
    """
    from pyhygrothermfem_fairyfly import _native
    nodes = [(index + 1, (index % 3) * 1.0, (index // 3) * 0.5)
             for index in range(9)]
    quads = [(1, 2, 5, 4), (2, 3, 6, 5), (4, 5, 8, 7), (5, 6, 9, 8)]
    materials = [(1.0, 0.9, 0.9, '', '', '', False, 1.0, 101325.0)] * 4
    film = tuple(10.0 + (index % 3) for index in range(1009))
    temperatures = tuple(20.0 - (index % 5) for index in range(1013))
    _t, _f, steps, slots = _native.solve_transient(
        nodes, quads, materials, [1.0] * 4, [1.0] * 4,
        convection=[(1, 2, film, temperatures),
                    (3, 6, (10.0,), (0.0,)),
                    (6, 9, (10.0,), (0.0,))],
        initial_temperature=10.0, initial_humidity=0.5,
        time_step=0.5, num_steps=200,
        convergence_tolerance=1e-8, relaxation=1.0)
    assert steps == 200
    assert slots == 1009 + 1013


def test_transient_periodic_bc_types_match_expanded_series():
    """Per-field periodic BCs reproduce the expanded-table semantics.

    For each varying BC kind (fixed temperature, heat flux, black-body
    radiation, linearized radiation), running the compact per-field
    series and running the lcm-expanded equivalents must give identical
    results -- the periodic forms are a memory optimization, not a
    semantic change.
    """
    from pyhygrothermfem_fairyfly import _native
    nodes = [(index + 1, (index % 3) * 1.0, (index // 3) * 0.5)
             for index in range(9)]
    quads = [(1, 2, 5, 4), (2, 3, 6, 5), (4, 5, 8, 7), (5, 6, 9, 8)]
    materials = [(1.0, 0.9, 0.9, '', '', '', False, 1.0, 101325.0)] * 4

    def run(**kwargs):
        frames = []
        _t, _f, steps, slots = _native.solve_transient(
            nodes, quads, materials, [1.0] * 4, [1.0] * 4,
            initial_temperature=10.0, initial_humidity=0.5,
            time_step=0.5, num_steps=14,
            convergence_tolerance=1e-8, relaxation=1.0,
            on_step=frames.append, **kwargs)
        assert steps == 14
        return frames, slots

    anchor = [(2, 3, (10.0,), (0.0,))]

    # Temperature pin: two-entry pin series vs its own expansion.
    frames, slots = run(
        convection=anchor,
        temperatures=[(1, 2, (30.0, 5.0))])
    assert slots == 2
    expanded, _slots = run(
        convection=anchor,
        temperatures=[(1, 2, (30.0, 5.0, 30.0, 5.0))])
    assert [f[2] for f in frames] == [f[2] for f in expanded]

    # Heat flux (2-period) vs its expansion; pin keeps the model anchored.
    frames, slots = run(
        convection=anchor,
        fluxes=[(1, 2, (4.0, -1.0))])
    assert slots == 2
    expanded, _slots = run(
        convection=anchor,
        fluxes=[(1, 2, (4.0, -1.0, 4.0, -1.0))])
    assert [f[2] for f in frames] == [f[2] for f in expanded]

    # Black-body radiation: emissivity period 2, temperature period 3.
    frames, slots = run(
        convection=anchor,
        radiations=[(1, 2, (0.9, 0.4), (20.0, 5.0, 15.0))])
    assert slots == 5
    expanded, _slots = run(
        convection=anchor,
        radiations=[(1, 2, (0.9, 0.4) * 3, (20.0, 5.0, 15.0) * 2)])
    assert [f[2] for f in frames] == [f[2] for f in expanded]

    # Linearized radiation: hr period 2, temperature period 3.
    frames, slots = run(
        convection=anchor,
        linearized_radiations=[(1, 2, (4.0, 2.0), (20.0, 5.0, 15.0))])
    assert slots == 5
    expanded, _slots = run(
        convection=anchor,
        linearized_radiations=[(1, 2, (4.0, 2.0) * 3, (20.0, 5.0, 15.0) * 2)])
    assert [f[2] for f in frames] == [f[2] for f in expanded]


def test_transient_periodic_temperature_pin_seeds_t0():
    """Periodic temperature pins seed the t=0 frame from series[0].

    Regression for the moved-from series read in the kernel:
    PeriodicTemperatureBC moved its PeriodicSeries into the convection
    coefficients and then read ``at(0)`` on the emptied series, which
    yields 0. The pinned nodes were re-seeded to 0 C, so the t=0 frame
    reported 0 C on a pin declared 30 C and the step-0 pin residual
    produced a ~3e19 W/m phantom heat rate on that edge. The fixed pin
    (single-element series) keeps TemperatureBC semantics throughout.
    """
    from pyhygrothermfem_fairyfly import _native
    nodes = [(index + 1, (index % 3) * 1.0, (index // 3) * 0.5)
             for index in range(9)]
    quads = [(1, 2, 5, 4), (2, 3, 6, 5), (4, 5, 8, 7), (5, 6, 9, 8)]
    materials = [(1.0, 0.9, 0.9, '', '', '', False, 1.0, 101325.0)] * 4
    anchor = [(2, 3, (10.0,), (0.0,))]

    def run(series):
        frames = []
        _t, _f, steps, _slots = _native.solve_transient(
            nodes, quads, materials, [1.0] * 4, [1.0] * 4,
            convection=anchor,
            temperatures=[(1, 2, series)],
            initial_temperature=10.0, initial_humidity=0.5,
            time_step=0.5, num_steps=3,
            convergence_tolerance=1e-8, relaxation=1.0,
            on_step=frames.append)
        assert steps == 3
        return frames

    periodic = run((30.0, 5.0))
    fixed = run((30.0,))
    # The t=0 frame reports the declared state: pinned nodes at 30 C,
    # matching the constant-pin seed (TemperatureBC), not a moved-from 0.
    assert periodic[0][2][0] == pytest.approx(30.0, abs=1e-9)
    assert periodic[0][2][0] == fixed[0][2][0]
    assert periodic[0][2][1] == pytest.approx(30.0, abs=1e-9)
    # Seeded at the pinned value, the step-0 pin residual vanishes: a
    # nonzero rate here is the phantom 1e18-film term of the old seed.
    pin_rates = [rate for a, b, rate in periodic[0][8]
                 if (a, b) == (0, 1)]
    assert pin_rates == pytest.approx([0.0] * len(pin_rates), abs=1e-9)
    assert pin_rates
    # Per-step pinning still cycles the series: step 0 pins 30 C,
    # step 1 pins 5 C on both edge nodes.
    assert periodic[1][2][:2] == pytest.approx([30.0, 30.0], abs=1e-9)
    assert periodic[2][2][:2] == pytest.approx([5.0, 5.0], abs=1e-9)
    # Interior nodes stay untouched by the seeding.
    assert periodic[0][2][2] == pytest.approx(10.0, abs=1e-9)


def test_transient_temperature_pin_t0_members(tmp_path):
    """End-to-end: a periodic temperature pin seeds the t=0 CSV row.

    Same moved-from seed read as above, seen through the THMZ members:
    ``Temperatures.csv`` row 0 must carry series[0] on the pinned face
    and ``HeatFluxEdges.csv`` row 0 must stay finite -- a phantom
    ~3e19 W/m pin rate is how the defect reaches the archive.
    """
    record = _transient_type_bc(
        bc_model='Dirichlet', use_temperature=True)
    record['convection'] = None
    timestep = {HOT_FILE: save_timestep_file(TimestepFile(
        name='Hot Air transient',
        temperature=({'index': 1, 'temperature': 30.0},
                     {'index': 2, 'temperature': 5.0})))}
    source = _transient_slab(
        tmp_path, name='pin-t0.thmz', type_records=[record],
        timestep_files=timestep, num_steps=4)
    result = solve(source)
    temperatures = _read_csv(
        result.members['transient results/Temperatures.csv'])
    edges = _read_csv(
        result.members['transient results/HeatFluxEdges.csv'])
    # Hot Air covers the left column (nodes 1, 4, 7); the pinned face
    # reports 30 C at t=0 and alternates 30/5 per step.
    t0 = temperatures['values'][0]
    assert [t0[i] for i in (0, 3, 6)] == pytest.approx(
        [30.0] * 3, abs=1e-9)
    assert temperatures['values'][1][0] == pytest.approx(30.0, abs=1e-9)
    assert temperatures['values'][2][0] == pytest.approx(5.0, abs=1e-9)
    # Pinned-edge rates at t=0 sit on the declared state: no phantom
    # 1e18-film residual may reach the archive.
    assert edges['values'][0][:2] == pytest.approx([0.0] * 2, abs=1e-9)


def test_transient_energy_balance_internal(tmp_path):
    """Boundary heat rates integrate to the internal-energy change.

    rho*cp = 1, slab volume 2 m3 (2x1 m face, unit depth): the time integral
    of the boundary edge rates plus stored energy must cancel to float
    accuracy (internal balance, not a THERM comparison).
    """
    result = solve(_transient_slab(tmp_path))
    members = result.members
    edges = _read_csv(members['transient results/HeatFluxEdges.csv'])
    temperatures = _read_csv(members['transient results/Temperatures.csv'])
    # Edge rates are outward-positive per edge; integrate with dt.
    net_outward = sum(
        sum(row) * TIME_STEP for row in edges['values'][1:])
    mean_t = [sum(row) / 9 for row in temperatures['values']]
    # 4 Q4 of 1 m x 0.5 m: volume 2 m3, rho*cp = 1 J/m3K.
    stored = (mean_t[-1] - mean_t[0]) * 2.0
    # Net outward flow plus stored energy must vanish (sign convention:
    # edge rates positive outward).
    assert abs(net_outward + stored) <= 1e-3 * max(1.0, abs(stored))


def test_transient_series_expand_cyclically(tmp_path):
    """A two-entry film series alternates per step (cyclic wrap)."""
    timestep = {HOT_FILE: save_timestep_file(TimestepFile(
        name='Hot Air transient',
        convection={'FixedConvectionFilmTimestep': _film_series(
            [20.0, 5.0])}))}
    alternating = solve(_transient_slab(
        tmp_path, name='alternating.thmz', timestep_files=timestep))
    constant = solve(_transient_slab(tmp_path, name='constant.thmz'))
    alternating_rows = _read_csv(
        alternating.members['transient results/Temperatures.csv'])
    constant_rows = _read_csv(
        constant.members['transient results/Temperatures.csv'])
    assert alternating_rows['times'] == constant_rows['times']
    # The driving series differs, so interior node histories must differ.
    left_column = [row[0] for row in alternating_rows['values'][1:]]
    reference_column = [row[0] for row in constant_rows['values'][1:]]
    assert any(a != pytest.approx(c, abs=1e-9)
               for a, c in zip(left_column, reference_column))
    # Both runs stay finite and bounded by the driving temperatures.
    assert all(-1e-9 <= value <= 20.0 + 1e-9
               for row in alternating_rows['values'] for value in row)


@pytest.mark.parametrize('form', ('empty', 'absent'))
def test_transient_steady_record_without_timestep_file(tmp_path, form):
    """bcType=SteadyState TransientData needs no timestep member.

    The writer emits an empty ``TimestepFileName`` for steady-type records;
    both that empty element and an omitted element must solve, taking the
    constant record fields (ticket 04 writer/solver seam).
    """
    record = _transient_type_bc(
        bc_type='SteadyState',
        convection={'model': 'Fixed Convection Coefficient',
                    'film_coefficient': 10.0, 'temperature': 20.0})

    def strip_file(members):
        model = ET.fromstring(members['Model.xml'])
        boundary = next(boundary for boundary in
                        model.findall('Boundaries/Boundary')
                        if boundary.findtext('ID') == '2')
        element = boundary.find('TransientData/TimestepFileName')
        if form == 'empty':
            element.text = ''
        else:
            boundary.find('TransientData').remove(element)
        members['Model.xml'] = ET.tostring(model)
        del members['timestep input files/' + HOT_FILE]

    source = _transient_slab(
        tmp_path, type_records=[record], members_hook=strip_file)
    result = solve(source)
    assert result.mode == 'dry-transient'
    assert list(result.temperatures) == pytest.approx(
        TRANSIENT_REF['steps'][-1], rel=1e-12, abs=1e-12)


def test_transient_timestep_halving_converges(tmp_path):
    """Halving dt over the same window moves the solution by < 1e-3 K.

    Internal timestep-convergence evidence (not a THERM comparison): the
    implicit scheme's first-order truncation must shrink under refinement.
    """
    coarse = solve(_transient_slab(tmp_path, name='coarse.thmz'))
    fine = solve(_transient_slab(
        tmp_path, name='fine.thmz',
        time_step=TIME_STEP / 2, num_steps=NUM_STEPS * 2))
    diff = max(abs(a - b) for a, b in zip(coarse.temperatures,
                                        fine.temperatures))
    assert diff < 1e-3
    assert diff > 0  # two grids of a first-order scheme do not coincide


def test_transient_steady_record_uses_constant_fields(tmp_path):
    """bcType=SteadyState records take their constant Convection fields."""
    record = _transient_type_bc(
        bc_type='SteadyState',
        convection={'model': 'Fixed Convection Coefficient',
                    'film_coefficient': 10.0, 'temperature': 20.0})
    source = _transient_slab(
        tmp_path, type_records=[record],
        timestep_files={HOT_FILE: save_timestep_file(
            TimestepFile(name='Hot Air transient'))})
    result = solve(source)
    assert result.mode == 'dry-transient'
    assert list(result.temperatures) == pytest.approx(
        TRANSIENT_REF['steps'][-1], rel=1e-12, abs=1e-12)


# --- explicit rejections -----------------------------------------------------

def test_steady_rejects_moisture_model(tmp_path):
    """SimulateMoisture without Transient mode stays refused (E5003)."""
    members = _slab_members()
    model = ET.fromstring(members['Model.xml'])
    model.find(
        'Properties/CalculationOptions/SimulateMoisture').text = 'true'
    members['Model.xml'] = ET.tostring(model)
    target = tmp_path / 'wet.thmz'
    with zipfile.ZipFile(target, 'w') as archive:
        for name, content in members.items():
            archive.writestr(name, content)
    with pytest.raises(UnsupportedError) as caught:
        solve(target)
    assert caught.value.code == 'THMZ-E5003'


def test_transient_rejects_humidity_boundary(tmp_path):
    """UseHumidity on a transient record is a declared-but-unsupported BC."""
    record = _transient_type_bc(use_humidity=True, humidity=0.5)
    timestep = {HOT_FILE: save_timestep_file(TimestepFile(
        name='Hot Air transient',
        convection={'FixedConvectionFilmTimestep': _film_series([20.0])},
        humidity=({'index': 1, 'humidity': 0.5},)))}
    source = _transient_slab(
        tmp_path, type_records=[record], timestep_files=timestep)
    with pytest.raises(UnsupportedError) as caught:
        solve(source)
    assert caught.value.code == 'THMZ-E5003'


def test_transient_rejects_montazeri_convection(tmp_path):
    """Montazeri declares at the data layer; the kernel has no BC for it."""
    record = _transient_type_bc(convection={'model': 'Montazeri'})
    timestep = {HOT_FILE: save_timestep_file(TimestepFile(
        name='Hot Air transient',
        convection={'MontazeriTimestep': (
            {'index': 1, 'temperature': 20.0, 'humidity': 0.5,
             'wind_speed': 3.0, 'wind_direction': 180.0},)}))}
    source = _transient_slab(
        tmp_path, type_records=[record], timestep_files=timestep)
    with pytest.raises(UnsupportedError) as caught:
        solve(source)
    assert caught.value.code == 'THMZ-E5001'


def test_transient_rejects_montazeri_on_steady_record(tmp_path):
    """bcType=SteadyState records must not bypass the model-level E5001.

    The steady-type early exit in input validation covers series
    requirements only; a declared Montazeri convection model is an
    unsupported-list rejection on every record, including constant-field
    records (before the fix this model fell through to a generic E5016).
    """
    record = _transient_type_bc(
        bc_type='SteadyState', convection={'model': 'Montazeri'})
    timestep = {HOT_FILE: save_timestep_file(
        TimestepFile(name='Hot Air transient'))}
    source = _transient_slab(
        tmp_path, type_records=[record], timestep_files=timestep)
    with pytest.raises(UnsupportedError) as caught:
        solve(source)
    assert caught.value.code == 'THMZ-E5001'


def test_transient_rejects_unsupported_convection_model(tmp_path):
    """TARP has a timestep schema but no transient kernel binding."""
    record = _transient_type_bc(convection={'model': 'TARP'})
    timestep = {HOT_FILE: save_timestep_file(TimestepFile(
        name='Hot Air transient',
        convection={'TARPTimestep': (
            {'index': 1, 'temperature': 20.0, 'humidity': 0.5},)}))}
    source = _transient_slab(
        tmp_path, type_records=[record], timestep_files=timestep)
    with pytest.raises(UnsupportedError) as caught:
        solve(source)
    assert caught.value.code == 'THMZ-E5016'


def test_transient_rejects_enclosure_radiation(tmp_path):
    """Automatic Enclosure radiation has no transient kernel binding."""
    record = _transient_type_bc(radiation={'model': 'Automatic Enclosure'})
    timestep = {HOT_FILE: save_timestep_file(TimestepFile(
        name='Hot Air transient',
        convection={'FixedConvectionFilmTimestep': _film_series([20.0])},
        radiation={'BlackBodyRadiationTimestep': (
            {'index': 1, 'temperature': 20.0, 'emissivity': 0.9},)}))}
    source = _transient_slab(
        tmp_path, type_records=[record], timestep_files=timestep)
    with pytest.raises(UnsupportedError) as caught:
        solve(source)
    assert caught.value.code == 'THMZ-E5016'


def test_transient_rejects_nonconstant_initial_conditions(tmp_path):
    """Only 'Constant everywhere' initial conditions are implemented."""
    source = _transient_slab(tmp_path, methodology='Use first timestep')
    with pytest.raises(UnsupportedError) as caught:
        solve(source)
    assert caught.value.code == 'THMZ-E5016'


def test_transient_requires_bulk_density_and_heat_capacity(tmp_path):
    """Transient materials need BulkDensity + SpecificHeatCapacityDry."""
    source = _transient_slab(tmp_path, density=False)
    with pytest.raises(ContractError) as caught:
        solve(source)
    assert caught.value.code == 'THMZ-E2004'


def test_transient_rejects_unanchored_model(tmp_path):
    """All-adiabatic transient models have no temperature anchor."""
    def strip_convection(members):
        bcs = ET.fromstring(members['SteadyStateBC.xml'])
        for record in bcs.findall('BoundaryCondition'):
            if record.findtext('Name') == 'Cold Air':
                record.find('Comprehensive/Convection/'
                            'FilmCoefficient').text = '0'
        members['SteadyStateBC.xml'] = ET.tostring(bcs)

    # The Hot Air transient record supplies h=0 at every step, so no
    # boundary anchors the temperature field.
    record = _transient_type_bc()
    timestep = {HOT_FILE: save_timestep_file(TimestepFile(
        name='Hot Air transient',
        convection={'FixedConvectionFilmTimestep': tuple(
            dict(entry, film_coefficient=0.0)
            for entry in _film_series([20.0]))}))}
    source = _transient_slab(
        tmp_path, type_records=[record], timestep_files=timestep,
        members_hook=strip_convection)
    with pytest.raises(ContractError) as caught:
        solve(source)
    assert caught.value.code == 'THMZ-E4006'


def test_transient_kernel_failure_is_solve_error(tmp_path, monkeypatch):
    from pyhygrothermfem_fairyfly import _native

    def fail_solve(*args, **kwargs):
        raise RuntimeError('transient kernel did not converge')

    monkeypatch.setattr(_native, 'solve_transient', fail_solve)
    with pytest.raises(SolveError) as caught:
        solve(_transient_slab(tmp_path))
    assert caught.value.code is None
