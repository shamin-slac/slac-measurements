import time

import slac_measurements.utils

from slac_measurements.wires.collection import BaseWireMeasurementCollection


class OTFWireMeasurementCollection(BaseWireMeasurementCollection):
    """Collect wire scan data using on-the-fly wire motion."""

    def _initialize_otf_with_retry(self, max_attempts: int = 3) -> None:
        """Start OTF scan and retry until wire is homed and on status."""

        # start_scan must always be called to arm the wire for OTF motion,
        # even if homed/on_status are already True from a prior run.
        for attempt in range(1, max_attempts + 1):
            self.logger.info(
                "Starting OTF scan on %s (Attempt %s/%s)...",
                self.beam_profile_device.name,
                attempt,
                max_attempts,
            )
            self.beam_profile_device.start_scan()

            if slac_measurements.utils.wait_until(
                lambda: self.beam_profile_device.homed
                and self.beam_profile_device.on_status
            ):
                self.logger.info("%s is homed and on.", self.beam_profile_device.name)
                return

            self.logger.warning(
                "%s did not become homed and on - retrying...",
                self.beam_profile_device.name,
            )

        raise RuntimeError(
            f"Failed to initialize {self.beam_profile_device.name} after {max_attempts} attempts."
        )

    def _run_collection_scan(self) -> None:
        """Run an OTF scan: init wire, start buffer."""

        def _start_timing_buffer() -> None:
            """Start BSA buffer and wait for completion while logging wire position."""

            self.logger.info("Starting buffer acquisition for on-the-fly scan...")
            acquisition_start = time.monotonic()
            acquisition_timeout_s = self._calculate_acquisition_timeout_s()
            self.my_buffer.start()

            time.sleep(0.5)

            i = 0
            while not self.my_buffer.is_acquisition_complete():
                elapsed_s = time.monotonic() - acquisition_start
                if elapsed_s > acquisition_timeout_s:
                    raise TimeoutError(
                        f"Timing buffer {self.my_buffer.number} did not complete after "
                        f"{elapsed_s:.1f}s (timeout={acquisition_timeout_s:.1f}s)."
                    )
                time.sleep(0.1)
                if i % 10 == 0:
                    wire_position = self.beam_profile_device.motor_rbv
                    self.logger.info("Wire position: %s", wire_position)
                i += 1

            self.logger.info(
                "Timing buffer %s acquisition complete after %.1f seconds",
                self.my_buffer.number,
                time.monotonic() - acquisition_start,
            )

        self.logger.info("Performing on-the-fly scan mode")
        self._initialize_otf_with_retry()
        _start_timing_buffer()
