"""Public t=0 recovery of a hand-worked wet Q4 field with varying gradient."""
from hashlib import sha256
import json
import math
import subprocess
import sys
import xml.etree.ElementTree as ET
import zipfile

import pytest

from ticket09_acceptance_cases import (
    PIN_UUID, file_digest, t0_observation, wet_slab)


EXPECTED_TEMPERATURES = (30.0, 10.0, 30.0, 30.0)
EXPECTED_FLUXES = (10.0, math.sqrt(500.0), 0.0, 20.0)
TEMPERATURE_ABS_K = 1e-9
FLUX_ABS_W_M2 = 1e-8


def _varying_gradient_input(path):
    """Add a matching top pin using the existing contracted boundary record."""
    wet_slab(path)
    with zipfile.ZipFile(path) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    model = ET.fromstring(members['Model.xml'])
    top = next(boundary for boundary in model.findall('Boundaries/Boundary')
               if boundary.findtext('ID') == '5')
    transient = ET.SubElement(top, 'TransientData')
    ET.SubElement(transient, 'UUID').text = PIN_UUID
    ET.SubElement(transient, 'TimestepFileName').text = 'ticket09-pin.xml'
    members['Model.xml'] = ET.tostring(model)
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, content in members.items():
            archive.writestr(name, content)
    return path


@pytest.mark.parametrize('entry', ('solve', 'run_thmz', 'cli'))
def test_wet_initial_heat_flux_matches_the_bilinear_field_at_every_node(
        tmp_path, entry):
    """Fourier flux is known before advancing this declared initial field.

    The nodal field T=30-10*x*(1-y) gives grad(T)=(-10*(1-y), 10*x).
    These derivatives are evaluated at the four rectangle corners, never
    inferred from a later solved frame or from the solver recovery routine.
    """
    from pyhygrothermfem_fairyfly import _native, run_thmz, solve

    physical_output = tmp_path
    source = _varying_gradient_input(physical_output / 'input.thmz')
    target = physical_output / 'output.thmz'
    frozen = {
        'entry': entry,
        'native_sha256': file_digest(_native.__file__),
        'driver_sha256': file_digest(__file__),
        'input': str(source), 'input_sha256': file_digest(source),
        'initial_field_c': 'T(x,y) = 30 - 10*x*(1-y)',
        'expected_node_ids': ['1', '2', '3', '4'],
        'expected_temperatures_c': EXPECTED_TEMPERATURES,
        'expected_flux_magnitudes_w_m2': EXPECTED_FLUXES,
        'temperature_tolerance_k': TEMPERATURE_ABS_K,
        'flux_tolerance_w_m2': FLUX_ABS_W_M2,
        'positive_time_step_s': 1.0, 'positive_steps': 2,
    }
    (physical_output / 'frozen-config.json').write_text(
        json.dumps(frozen, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    if entry == 'solve':
        members = solve(source).members
        with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as archive:
            for name, content in members.items():
                archive.writestr(name, content)
    elif entry == 'run_thmz':
        run_thmz(source, output=target)
    else:
        process = subprocess.run([
            sys.executable, '-I', '-B', '-m', 'pyhygrothermfem_fairyfly', 'run-thmz',
            str(source), '--output', str(target)], cwd=physical_output,
            capture_output=True, text=True, encoding='utf-8',
            errors='replace', timeout=60)
        (physical_output / 'cli.stdout.log').write_text(
            process.stdout, encoding='utf-8')
        (physical_output / 'cli.stderr.log').write_text(
            process.stderr, encoding='utf-8')
        assert process.returncode == 0
    with zipfile.ZipFile(target) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    observed = t0_observation(members)
    error = max(abs(actual - expected) for actual, expected in zip(
        observed['HeatFlux']['values'], EXPECTED_FLUXES))
    temperature_error = max(abs(actual - expected) for actual, expected in zip(
        observed['Temperatures']['values'], EXPECTED_TEMPERATURES))
    provenance = json.loads(members['pyHygroTherm.json'])
    (physical_output / 'observation.json').write_text(json.dumps({
        'frozen': frozen, 'observed_t0': observed,
        'maximum_heat_flux_error_w_m2': error,
        'maximum_temperature_error_k': temperature_error,
        'output': str(target), 'output_sha256': file_digest(target),
        'member_sha256': {name: sha256(content).hexdigest()
                          for name, content in members.items()},
        'input_unchanged': file_digest(source) == frozen['input_sha256'],
        'provenance': provenance,
        'status': 'pass' if error <= FLUX_ABS_W_M2
        and temperature_error <= TEMPERATURE_ABS_K else 'fail',
    }, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    assert provenance['input_sha256'] == frozen['input_sha256']
    assert file_digest(source) == frozen['input_sha256']
    assert observed['Temperatures']['entities'] == ['1', '2', '3', '4']
    assert observed['Temperatures']['values'] == pytest.approx(
        EXPECTED_TEMPERATURES, abs=TEMPERATURE_ABS_K, rel=0)
    assert observed['HeatFlux']['values'] == pytest.approx(
        EXPECTED_FLUXES, abs=FLUX_ABS_W_M2, rel=0)
