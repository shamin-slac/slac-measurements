from typing import Optional

import numpy as np
from pydantic import model_validator
from slac_devices.beampath import Beampath
from slac_devices.reader import create_beampath
from slac_devices.wire import Wire
from slac_measurements.measurement import Measurement
from slac_timing import Buffer


class TMITLoss(Measurement):
    """Measures percentage beam intensity loss across a wire scanner."""

    name: str = "TMIT Loss"
    buffer: Buffer
    beampath: str
    beam_profile_device: Wire

    _beampath_obj: Optional[Beampath] = None
    bpms: Optional[dict] = None
    idx_upstream: Optional[list] = None
    idx_downstream: Optional[list] = None

    @model_validator(mode="after")
    def _setup_bpms(self) -> "TMITLoss":
        """Load BPMs from beampath and resolve upstream/downstream indices."""
        self._beampath_obj = create_beampath(self.beampath, device_types={"bpms"})
        all_bpms = self._beampath_obj.bpms
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
        n_samples = self.buffer.n_measurements
        bpm_names = list(self.bpms.keys())

        all_data = {}
        for area in self._beampath_obj.areas.values():
            if area.bpm_collection:
                all_data.update(
                    area.bpm_collection.get_buffer_data(self.buffer, pad=True)
                )

        rows = [
            np.asarray(all_data[name], dtype=float)
            if name in all_data
            else np.full(n_samples, np.nan)
            for name in bpm_names
        ]
        return np.array(rows)

    @staticmethod
    def _calc_tmit_loss(
        data: np.ndarray, idx_upstream: list, idx_downstream: list
    ) -> np.ndarray:
        """Normalize TMIT data and compute percentage loss between upstream/downstream BPMs."""
        row_medians = np.nanmedian(data, axis=1, keepdims=True)
        ironed = data / row_medians

        mean_iron_upstream = np.nanmean(ironed[idx_upstream], axis=0)
        normed = ironed / mean_iron_upstream

        mean_upstream = np.nanmean(normed[idx_upstream], axis=0)
        mean_downstream = np.nanmean(normed[idx_downstream], axis=0)

        return (mean_upstream - mean_downstream) * 100
