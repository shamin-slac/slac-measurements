from typing import Optional

import epics
import numpy as np
from pydantic import model_validator
from slac_devices.reader import create_beampath
from slac_devices.wire import Wire
from slac_measurements.measurement import Measurement
from edef import BSABuffer, EventDefinition


class TMITLoss(Measurement):
    """Measures percentage beam intensity loss across a wire scanner."""

    name: str = "TMIT Loss"
    buffer: BSABuffer | EventDefinition
    beampath: str
    beam_profile_device: Wire

    bpms: Optional[dict] = None
    idx_upstream: Optional[list] = None
    idx_downstream: Optional[list] = None

    @model_validator(mode="after")
    def _setup_bpms(self) -> "TMITLoss":
        """Create BPMs at construction time so PVs can connect before measure()."""
        beampath_obj = create_beampath(self.beampath, device_types={"bpms"})
        all_bpms = beampath_obj.bpms
        if not all_bpms:
            raise LookupError("No BPMs found in beampath.")
        self.bpms = dict(sorted(all_bpms.items(), key=lambda item: item[1].z_location))

        bpms_upstream = self.beam_profile_device.metadata.tmitloss.upstream
        bpms_downstream = self.beam_profile_device.metadata.tmitloss.downstream
        bpm_names = list(self.bpms.keys())
        self.idx_upstream = [
            bpm_names.index(name) for name in bpms_upstream if name in self.bpms
        ]
        self.idx_downstream = [
            bpm_names.index(name) for name in bpms_downstream if name in self.bpms
        ]
        return self

    def measure(self):
        """Acquire TMIT data and return percentage loss as a numpy array."""
        data = self._get_bpm_data()
        return self._calc_tmit_loss(data, self.idx_upstream, self.idx_downstream)

    def _get_bpm_data(self) -> np.ndarray:
        """Collect TMIT buffer data from all BPMs. Returns shape (n_bpms, n_samples)."""
        pv_names = [
            f"{bpm.controls_information.control_name}:TMIT"
            for bpm in self.bpms.values()
        ]
        hst_pvs = [f"{pv}HST{self.buffer.number}" for pv in pv_names]
        results = epics.caget_many(hst_pvs)

        n_samples = self.buffer.n_measurements
        rows = []
        for name, data in zip(self.bpms.keys(), results):
            if data is None or len(data) < n_samples:
                if data is not None:
                    print(
                        f"Skipping BPM {name}: incomplete data ({len(data)}/{n_samples})"
                    )
                else:
                    print(f"Skipping BPM {name}: no buffer data")
                row = np.full(n_samples, np.nan)
            else:
                row = np.asarray(data[:n_samples], dtype=float)
            rows.append(row)
        return np.array(rows)

    @staticmethod
    def _calc_tmit_loss(
        data: np.ndarray, idx_upstream: list, idx_downstream: list
    ) -> np.ndarray:
        """Normalize TMIT data and compute percentage loss between before/after wire BPMs."""
        row_medians = np.nanmedian(data, axis=1, keepdims=True)
        ironed = data / row_medians

        mean_iron_upstream = np.nanmean(ironed[idx_upstream], axis=0)
        normed = ironed / mean_iron_upstream

        mean_upstream = np.nanmean(normed[idx_upstream], axis=0)
        mean_downstream = np.nanmean(normed[idx_downstream], axis=0)

        return (mean_upstream - mean_downstream) * 100
