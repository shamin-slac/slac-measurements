import time

from slac_measurements.wires.collection import BaseWireMeasurementCollection


class OTFWireMeasurementCollection(BaseWireMeasurementCollection):
    """Collect wire scan data using on-the-fly wire motion."""

    def _run_collection_scan(self) -> None:
        """Run an OTF scan: init wire, start buffer."""

        def _start_timing_buffer() -> None:
            """Start BSA buffer and wait for completion while logging wire position."""

            self.logger.info("Starting buffer acquisition for on-the-fly scan...")
            acquisition_start = time.monotonic()
            self.my_buffer.start()

            time.sleep(0.5)

            i = 0
            while not self.my_buffer.is_acquisition_complete():
                time.sleep(0.1)
                if i % 10 == 0:
                    wire_position = self.my_wire.motor_rbv
                    self.logger.info("Wire position: %s", wire_position)
                i += 1

            self.logger.info(
                "Timing buffer %s acquisition complete after %.1f seconds",
                self.my_buffer.number,
                time.monotonic() - acquisition_start,
        )

        self.logger.info("Performing on-the-fly scan mode")
        self._initialize_wire_with_retry(scan_mode="otf")
        _start_timing_buffer()
