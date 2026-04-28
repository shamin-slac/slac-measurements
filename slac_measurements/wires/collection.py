import logging
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import model_validator
from typing_extensions import Self

from slac_devices.wire import Wire
import slac_measurements.beam_profile
import slac_measurements.wires.buffer
import slac_measurements.utils
from slac_measurements.wires.collection_results import (
    MeasurementMetadata,
    WireMeasurementCollectionResult,
)

_LOG_DIR = Path("/u1/lcls/physics/data/wire_scan/logs")
_LOGGER_NAME = "wire_scan_logger"
ScanMode = Literal["step", "otf"]


class BaseWireMeasurementCollection(
    slac_measurements.beam_profile.BeamProfileMeasurement,
    ABC,
):
    """
    Collects wire scan measurement data via motor motion and timing buffer.
    Raw data is returned for downstream analysis.

    Attributes:
        beam_profile_device (Wire): Wire device for the scan.
        beampath (str): Beamline identifier for buffer and device selection.
        my_buffer: Timing buffer managing data acquisition.
        devices (dict): Device objects (wire, detectors) used in the scan.
        data (dict): Raw buffered data by device name.
        logger (logging.Logger): File-based measurement logger.
    """

    name: str = "Wire Beam Profile Measurement"
    beam_profile_device: Wire
    beampath: str
    my_buffer: object | None = None
    devices: dict | None = None
    detectors: list | None = None
    data: dict | None = None
    logger: logging.Logger | None = None
    metadata: MeasurementMetadata | None = None

    def measure(self) -> WireMeasurementCollectionResult:
        """
        Execute wire scan: run mode-specific wire motion and acquire detector
        data from timing buffer.

        Returns
        -------
        WireMeasurementCollectionResult
            Raw data and metadata, including:
            - raw_data: Buffered position and detector values by device name
            - metadata: Timestamp, wire name, area, beampath, and detector list
        """

        def _release_buffer_safely() -> None:
            """Release timing buffer after scan completion."""

            buf = self.my_buffer
            if buf is not None:
                buffer_number = getattr(buf, "number", None)
                try:
                    self.logger.info("Releasing timing buffer %s.", buffer_number)
                    buf.release()
                except Exception:
                    self.logger.exception("Failed while releasing timing buffer %s.", buffer_number)
                finally:
                    self.my_buffer = None

        self._prepare_runtime_state()

        try:
            self._run_collection_scan()
            self.data = self._get_data_from_buffer()
            self.metadata.timestamp = datetime.now()
        finally:
            _release_buffer_safely()

        return WireMeasurementCollectionResult(
            raw_data=self.data,
            metadata=self.metadata,
        )

    # alias so beam_profile_device can also be accessed with name my_wire
    @property
    def my_wire(self) -> Wire:
        return self.beam_profile_device

    @my_wire.setter
    def my_wire(self, value):
        self.beam_profile_device = value

    def _create_device_dictionary(self) -> dict:
        """Create dictionary of required devices. Includes the wire device and detectors."""

        def _instantiate_device(name: str, area: str):
            """Instantiate a single device by name and area."""

            import slac_devices.reader
            import slac_measurements.tmit_loss

            if name == "TMITLOSS":
                return slac_measurements.tmit_loss.TMITLoss(
                    my_buffer=self.my_buffer,
                    my_wire=self.my_wire,
                    beampath=self.beampath,
                    region=self.my_wire.area,
                )

            create_by_prefix = {
                "LBLM": slac_devices.reader.create_lblm,
                "PMT": slac_devices.reader.create_pmt,
            }

            creator = next(
                (f for prefix, f in create_by_prefix.items() if name.startswith(prefix)),
                None,
            )

            if creator is None:
                self.logger.warning("Unknown device type '%s'. Skipping.", name)
                return None

            device = creator(area=area, name=name)
            if device is None:
                self.logger.warning("Device creation for %s returned None.", name)

            return device

        self.logger.info("Creating device dictionary...")

        devices = {self.my_wire.name: self.my_wire}

        for ds in self.my_wire.metadata.detectors:
            name, area = ds.split(":")
            detector = _instantiate_device(name, area)
            if detector is not None:
                devices[name] = detector

        self.logger.info("Device dictionary built.")
        return devices

    def _create_metadata(self) -> MeasurementMetadata:
        """Create per-run metadata for the current scan."""

        def _get_default_detector() -> str:
            """Determine the default detector for analysis from wire metadata or device list."""

            default_detector = self.my_wire.metadata.default_detector

            if not default_detector:
                if not self.detectors:
                    msg = (
                        "No detectors available from wire metadata; "
                        "cannot determine default detector."
                    )
                    self.logger.error(msg)
                    raise RuntimeError(msg)
                return self.detectors[0]

            return default_detector.split(":", 1)[0]

        def _get_scan_ranges() -> dict:
            """Return dictionary of scan ranges for x, y, and u motors."""

            return {
                "x": self.my_wire.x_range,
                "y": self.my_wire.y_range,
                "u": self.my_wire.u_range,
            }

        return MeasurementMetadata(
            wire_name=self.my_wire.name,
            buffer_number=self.my_buffer.number,
            area=self.my_wire.area,
            beampath=self.beampath,
            detectors=self.detectors,
            default_detector=_get_default_detector(),
            scan_ranges=_get_scan_ranges(),
            timestamp=None,
            active_profiles=self.my_wire.active_profiles(),
            install_angle=self.my_wire.install_angle,
            notes=None,
        )

    def _get_data_from_buffer(self) -> dict:
        """Collects wire scan and detector data after buffer completes."""

        def _get_buffer_collection_method(device_name: str) -> str | None:
            """Determine the buffer collection method for a given device based on its name."""

            if device_name == self.my_wire.name:
                return "position_buffer"
            elif device_name.startswith("LBLM"):
                return "fast_buffer"
            elif device_name.startswith("PMT"):
                return "qdcraw_buffer"
            else:
                return None

        def _collect_device_data(device_name: str) -> np.ndarray:
            """Collect data for a given device using the appropriate method."""

            device = self.devices[device_name]
            buffer_method = _get_buffer_collection_method(device_name)

            if buffer_method is None:
                return (
                    device.measure()
                )  # For devices like TMITLOSS that don't use buffer collection

            return slac_measurements.utils.collect_with_size_check(
                device, buffer_method, self.my_buffer, self.logger
            )

        self.logger.info("Getting data from timing buffer ...")
        data = {name: _collect_device_data(name) for name in self.devices.keys()}
        self.logger.info("Data retrieved from buffer. Scan complete.")
        return data

    def _initialize_wire_with_retry(self,
        scan_mode: ScanMode,
        max_attempts: int = 3,
    ) -> None:
        """Initialize wire motion with retries until wire is ready for scan mode.

        Readiness conditions:
        - otf: wait for both my_wire.homed and my_wire.on_status.
        - step: wait for my_wire.enabled.

        scan_mode must be 'otf' or 'step'; raises on failure.
        """

        if scan_mode == "otf":
            action_method = self.my_wire.start_scan
            ready_check = lambda: self.my_wire.homed and self.my_wire.on_status
            ready_desc = "homed and on"
            action_desc = "for on-the-fly scan"
        elif scan_mode == "step":
            action_method = self.my_wire.initialize
            ready_check = lambda: self.my_wire.enabled
            ready_desc = "enabled"
            action_desc = "for step scan"

        # Skip initialization if wire is already in the expected ready state
        if ready_check():
            self.logger.info("%s is already %s.", self.my_wire.name, ready_desc)
            return

        for attempt in range(1, max_attempts + 1):
            self.logger.info(
                "Initializing %s %s: (Attempt %s/%s)...",
                self.my_wire.name, action_desc, attempt, max_attempts,
            )
            action_method()

            # If returns True within timeout, proceed
            if slac_measurements.utils.wait_until(ready_check):
                self.logger.info(
                    "%s initialized (%s is True).", self.my_wire.name, ready_desc
                )
                return

            # After timeout, log and iterate through for loop again
            else:
                self.logger.warning(
                    "%s did not enable - retrying...", self.my_wire.name
                )

        raise RuntimeError(
            f"Failed to initialize {self.my_wire.name} after {max_attempts} attempts."
        )

    def _prepare_runtime_state(self) -> None:
        """Prepare per-run state that depends on an active timing buffer."""

        self.my_buffer = self._reserve_buffer()
        self.devices = self._create_device_dictionary()
        self.metadata = self._create_metadata()

    def _reserve_buffer(self) -> object:
        """Reserve a timing buffer for the scan based on beampath and wire metadata."""

        if self.my_buffer is None:
            self.my_buffer = slac_measurements.wires.buffer.reserve_buffer(
                beampath=self.beampath,
                logger=self.logger,
                pulses=self.my_wire.scan_pulses,
                beam_rate=self.my_wire.beam_rate,
            )

        return self.my_buffer

    @abstractmethod
    def _run_collection_scan(self) -> None:
        """Run mode-specific wire motion and buffer timing behavior."""

    @model_validator(mode="after")
    def _run_setup(self) -> Self:
        """Initialize construction-time state for a collection instance."""

        import slac_measurements.logger.file_logger

        # Configure logger — compute filename now so long-running processes
        # get a fresh date-stamped file rather than the one frozen at import.
        log_filepath = _LOG_DIR / f"ws_log_{datetime.now().strftime('%Y%m%d')}.txt"
        self.logger = slac_measurements.logger.file_logger.custom_logger(
            log_file=str(log_filepath),
            name=_LOGGER_NAME,
        )
        self.logger.propagate = False

        # Get list of detector names from wire metadata
        self.detectors = [d.split(":")[0] for d in self.my_wire.metadata.detectors]
        return self

def create_wire_collection(*,
    scan_mode: ScanMode,
    beam_profile_device: Wire,
    beampath: str,
) -> BaseWireMeasurementCollection:
    """Instantiate the mode-specific wire collection class."""

    if scan_mode == "step":
        from slac_measurements.wires.step_collection import StepWireMeasurementCollection

        return StepWireMeasurementCollection(
            beam_profile_device=beam_profile_device,
            beampath=beampath,
        )

    if scan_mode == "otf":
        from slac_measurements.wires.otf_collection import OTFWireMeasurementCollection

        return OTFWireMeasurementCollection(
            beam_profile_device=beam_profile_device,
            beampath=beampath,
        )

    raise ValueError(f"Unknown scan_mode '{scan_mode}'. Expected 'step' or 'otf'.")