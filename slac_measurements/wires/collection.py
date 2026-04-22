import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Literal, Optional

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

_DATE = datetime.now().strftime("%Y%m%d")
_LOG_FILENAME = f"ws_log_{_DATE}.txt"
_LOGGER_NAME = "wire_scan_logger"
ScanMode = Literal["step", "otf"]


class BaseWireMeasurementCollection(
    slac_measurements.beam_profile.BeamProfileMeasurement,
    ABC,
):
    """
    Collects wire scan measurement data via motor motion and BSA buffer.

    Moves the wire and acquires synchronized detector data without organizing
    or fitting. Raw data is returned for downstream analysis.

    Attributes:
        beam_profile_device (Wire): Wire device for the scan.
        beampath (str): Beamline identifier for buffer and device selection.
        my_buffer: BSA buffer managing data acquisition.
        devices (dict): Device objects (wire, detectors) used in the scan.
        data (dict): Raw buffered data by device name.
        logger (logging.Logger): File-based measurement logger.
    """

    name: str = "Wire Beam Profile Measurement"
    beam_profile_device: Wire
    beampath: str
    my_buffer: Optional[object] = None
    devices: Optional[dict] = None
    detectors: Optional[list] = None
    data: Optional[dict] = None
    logger: Optional[logging.Logger] = None

    # alias so beam_profile_device can also be accessed with name my_wire
    @property
    def my_wire(self) -> Wire:
        return self.beam_profile_device

    @my_wire.setter
    def my_wire(self, value):
        self.beam_profile_device = value

    def measure(self) -> WireMeasurementCollectionResult:
        """
        Execute wire scan: run mode-specific wire motion and acquire detector
        data from BSA buffer.

        Returns
        -------
        WireMeasurementCollectionResult
            Raw data and metadata, including:
            - raw_data: Buffered position and detector values by device name
            - metadata: Timestamp, wire name, area, beampath, and detector list
        """
        self.my_buffer = self._reserve_buffer()
        metadata = self._create_metadata()

        try:
            self._run_collection_scan()
            self.data = self._get_data_from_buffer()
        finally:
            self._release_buffer_safely()

        return WireMeasurementCollectionResult(
            raw_data=self.data,
            metadata=metadata,
        )

    @model_validator(mode="after")
    def run_setup(self) -> Self:
        import slac_measurements.logger.file_logger
        # Configure  logger
        self.logger = slac_measurements.logger.file_logger.custom_logger(
            log_file=_LOG_FILENAME,
            name=_LOGGER_NAME,
        )
        self.logger.propagate = False

        # Reserve BSA buffer
        self.my_buffer = self._reserve_buffer()

        # Get list of detector names from wire metadata
        self.detectors = [d.split(":")[0] for d in self.my_wire.metadata.detectors]

        # Generate dictionary of all required lcls-tools device objects
        self.devices = self._create_device_dictionary()
        return self

    def _create_device_dictionary(self) -> dict:
        """
        Creates a device dictionary for a wire scan setup.  Includes the wire
        device and any associated detectors from metadata.

        Returns:
            dict: A mapping of device names to device objects.
        """

        def _instantiate_device(name: str, area: str):
            """
            Instantiate a single device by name and area
            """
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

        # Instantiate device dictionary with wire device
        devices = {self.my_wire.name: self.my_wire}

        # ds is a colon-separated detector string from metadata
        # e.g. "LBLM:TEST" -> name = "LBLM", area = "TEST"
        for ds in self.my_wire.metadata.detectors:
            name, area = ds.split(":")
            detector = _instantiate_device(name, area)
            if detector is not None:
                devices[name] = detector

        self.logger.info("Device dictionary built.")
        return devices

    def _create_metadata(self) -> MeasurementMetadata:
        """
        Make additional metadata.
        """
        def _get_default_detector() -> str:
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

            # Metadata may be stored as "<name>:<area>"; analysis expects the device name key.
            return default_detector.split(":", 1)[0]

        def _get_scan_ranges():
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
            timestamp=datetime.now(),
            active_profiles=self.my_wire.active_profiles(),
            install_angle=self.my_wire.install_angle,
            notes=None,
        )

    def _get_data_from_buffer(self) -> dict:
        """
        Collects wire scan and detector data after buffer completes.

        Returns:
            dict: Collected data keyed by device name.
        """
        def _get_buffer_collection_method(device_name: str) -> Optional[str]:
            """
            Determine the buffer collection method for a given device based on its name.
            Returns None for devices that don't collect data this way (e.g., TMITLOSS).
            """
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
            import slac_measurements.utils

            device = self.devices[device_name]
            buffer_method = _get_buffer_collection_method(device_name)

            if buffer_method is None:
                return (
                    device.measure()
                )  # For devices like TMITLOSS that don't use buffer collection

            return slac_measurements.utils.collect_with_size_check(
                device, buffer_method, self.my_buffer, self.logger
            )

        self.logger.info("Getting data from BSA buffer...")
        data = {name: _collect_device_data(name) for name in self.devices.keys()}
        self.logger.info("Data retrieved from buffer. Scan complete.")
        return data

    def _initialize_wire_with_retry(
        self,
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
            self.logger.info(f"{self.my_wire.name} is already {ready_desc}.")
            return

        for attempt in range(1, max_attempts + 1):
            self.logger.info(
                f"Initializing {self.my_wire.name} {action_desc}: "
                f"(Attempt {attempt}/{max_attempts})..."
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

    def _release_buffer_safely(self) -> None:
        """Release BSA resources after scan completion."""
        if self.my_buffer is not None:
            try:
                self.logger.info("Releasing BSA buffer.")
                self.my_buffer.release()
            except Exception:
                self.logger.exception("Failed while releasing BSA buffer.")
            finally:
                self.my_buffer = None

    def _reserve_buffer(self) -> object:
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

def create_wire_collection(
    *,
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