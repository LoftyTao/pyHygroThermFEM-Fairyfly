"""The public THMZ path records a real solved energy error when requested."""
import json
from pathlib import Path
import tempfile
from xml.etree import ElementTree as ET
from zipfile import ZipFile, ZIP_DEFLATED

from pyhygrothermfem_fairyfly import run_thmz


def test_energy_error_is_calculated_only_when_requested():
    slab = Path(__file__).parent / 'assets' / 'slab.thmz'
    with tempfile.TemporaryDirectory() as folder:
        checked = Path(run_thmz(slab, Path(folder) / 'checked.thmz'))
        with ZipFile(checked) as archive:
            provenance = json.loads(archive.read('pyHygroTherm.json'))
        assert provenance['energy_error_method'] == (
            'HygroThermFEM recovery energy norm')
        assert 0 <= provenance['energy_error_percent'] < 1e-6

        with ZipFile(slab) as archive:
            members = {name: archive.read(name) for name in archive.namelist()}
        model = ET.fromstring(members['Model.xml'])
        model.find('Properties/CalculationOptions/MeshControl/'
                   'RunErrorEstimator').text = 'false'
        members['Model.xml'] = ET.tostring(model, encoding='utf-8')
        disabled = Path(folder) / 'disabled.thmz'
        with ZipFile(disabled, 'w', ZIP_DEFLATED) as archive:
            for name, content in members.items():
                archive.writestr(name, content)
        run_thmz(disabled, disabled)
        with ZipFile(disabled) as archive:
            provenance = json.loads(archive.read('pyHygroTherm.json'))
        assert provenance['energy_error_percent'] is None
        assert provenance['energy_error_method'] is None
