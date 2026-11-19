import time
from typing import Dict, Union

import numpy as np

from slac_devices.device import Device
from slac_devices.magnet import Magnet

from slac_devices.screen import Screen
from slac_devices.wire import Wire


def quad_scan_optics(
    magnet: Magnet, beam_profile_device: Union[Screen, Wire], physics_model="BMAD"
) -> Dict:
    """Get rmat (6 x 6) from magnet to measurement device;
    get design twiss and energy at beginning of magnet"""
    rmat = get_rmat(magnet, beam_profile_device, physics_model)
    twiss = get_twiss(magnet, physics_model, True, "beg")
    energy = get_energy(magnet, physics_model, False, "beg")
    return {"rmat": rmat, "design_twiss": twiss, "energy": energy}


def get_optics_after_magnet(
    magnet: Magnet, beam_profile_device: Union[Screen, Wire], physics_model="BMAD"
) -> Dict:
    """Get rmat (6 x 6) from end of magnet to measurement device;
    get design twiss and energy at beginning of magnet"""
    full_rmat = get_rmat(magnet, beam_profile_device, physics_model)
    quad_rmat = get_rmat(magnet, magnet, physics_model)
    after_quad_rmat = full_rmat @ np.linalg.inv(quad_rmat)
    twiss = get_twiss(magnet, physics_model, True, "beg")
    energy = get_energy(magnet, physics_model, False, "beg")
    return {"after_quad_rmat": after_quad_rmat, "design_twiss": twiss, "energy": energy}


def multi_device_optics(
    beam_profile_devices: list[Union[Screen, Wire]], physics_model="BMAD"
) -> Dict:
    """Get rmat from reference device to all measurement devices (n x 6 x 6);
    get design twiss and energy at reference device"""
    rmat = []
    ref_index = int(len(beam_profile_devices) / 2)
    device_ref = beam_profile_devices[ref_index]
    for device in beam_profile_devices:
        rmat.append(get_rmat(device_ref, device, physics_model))
    rmat = np.array(rmat)
    twiss = get_twiss(device_ref, physics_model, True)
    energy = get_energy(device_ref, physics_model, False)
    return {"rmat": rmat, "design_twiss": twiss, "energy": energy}


def get_rmat(
    device1: Device,
    device2: Device,
    physics_model="BMAD",
    use_design=False,
    device1_pos="beg",
    device2_pos="end",
):
    """Get rmat from device1 to device2"""
    if physics_model == "BLEM":
        refresh_blem_model()
    else:
        time.sleep(2)
    model = _get_model_from_device(device1, physics_model, use_design=use_design)
    rmat = model.get_rmat(
        device1.name,
        device2.name,
        from_device_pos=device1_pos,
        to_device_pos=device2_pos,
    )

    return rmat


def get_twiss(device: Device, physics_model="BMAD", use_design=True, pos="mid"):
    """Get twiss at device"""
    if physics_model == "BLEM":
        refresh_blem_model()
    elif physics_model == "BMAD":
        time.sleep(2)
    model = _get_model_from_device(device, physics_model, use_design=use_design)
    twiss = model.get_twiss(device.name, pos=pos)

    return twiss


def get_energy(device: Device, physics_model="BMAD", use_design=False, pos="mid"):
    """Get energy at device"""
    if physics_model == "BLEM":
        refresh_blem_model()
    elif physics_model == "BMAD":
        time.sleep(2)
    model = _get_model_from_device(device, physics_model, use_design=use_design)
    energy = model.get_twiss(device.name, pos=pos)["p0c"] / 1e9  # GeV

    return energy


def refresh_blem_model():
    from epics import PV
    import threading

    done = threading.Event()

    def on_change(pvname=None, value=None, **kwargs):
        if value == 0:
            done.set()

    # writing 1 to model ctrl PV causes BLEM model to update
    model_ctrl_pv = PV("BLEM:SYS0:1:MAT_MODEL:CTRL")
    model_ctrl_pv.add_callback(on_change)
    model_ctrl_pv.put(1, wait=True)  # blocks until write has processed
    done.wait()  # blocks until ctrl PV has reset to 0


def _get_model_from_device(device, physics_model, use_design=False):
    from meme.model import Model

    # Look for device beam path in meme beam paths
    beam_path = None
    for bp in device.metadata.beam_path:
        if bp in [
            "CU_HXR",
            "CU_SXR",
            "CU_SPEC",
            "SC_DIAG0",
            "SC_BSYD",
            "SC_HXR",
            "SC_SXR",
            "FACET2E",
        ]:
            beam_path = bp
            break
    if beam_path is None:
        raise ValueError("Valid beam path not found in device metadata.")

    return Model(beam_path, model_source=physics_model, use_design=use_design)
