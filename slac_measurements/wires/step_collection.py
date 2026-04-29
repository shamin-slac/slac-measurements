import time

import slac_measurements.utils

from slac_measurements.wires.collection import BaseWireMeasurementCollection

_WIRE_TOLERANCE = 250  # microns
_WIRE_RETRACT_WAIT = 2  # seconds


class StepWireMeasurementCollection(BaseWireMeasurementCollection):
    """Collect wire scan data using discrete step motion."""

    def _run_collection_scan(self) -> None:
        """Run a step scan: init wire, start buffer, move positions, retract, wait."""

        def _calculate_step_speed(position_index: int,
                                  positions: list[int]) -> int:
            """Return speed for a step position: max for inner, computed for outer."""

            if position_index % 2 == 0:
                return int(self.my_wire.speed_max)

            position_delta = positions[position_index] - positions[position_index - 1]
            speed = (position_delta / self.my_wire.scan_pulses) * self.my_wire.beam_rate
            return int(speed)

        def _get_step_positions() -> list[int]:
            """Return sorted inner and outer positions for active profiles."""

            positions = []
            for profile in self.my_wire.active_profiles():
                for mode in ["inner", "outer"]:
                    attr_name = f"{profile}_wire_{mode}"
                    positions.append(getattr(self.my_wire, attr_name))
            return sorted(positions)

        def _initialize_step_with_retry(max_attempts: int = 3) -> None:
            """Initialize wire for step scan mode with retries until enabled."""

            # initialize is idempotent — skip if wire is already enabled.
            if self.my_wire.enabled:
                self.logger.info("%s is already enabled.", self.my_wire.name)
                return

            for attempt in range(1, max_attempts + 1):
                self.logger.info(
                    "Initializing %s for step scan (Attempt %s/%s)...",
                    self.my_wire.name,
                    attempt,
                    max_attempts,
                )
                self.my_wire.initialize()

                if slac_measurements.utils.wait_until(lambda: self.my_wire.enabled):
                    self.logger.info("%s initialized (enabled is True).", self.my_wire.name)
                    return

                self.logger.warning("%s did not enable - retrying...", self.my_wire.name)

            raise RuntimeError(
                f"Failed to initialize {self.my_wire.name} after {max_attempts} attempts."
            )

        def _move_to_step_position(*,
            position: int,
            position_index: int,
            total_positions: int,
            positions: list[int],
        ) -> None:
            """Move wire to a step position and wait for arrival."""

            self.logger.info(
                "Moving wire to %s (step %s/%s)...",
                position,
                position_index + 1,
                total_positions,
            )

            self.my_wire.speed = _calculate_step_speed(position_index, positions)
            self.my_wire.motor = position

            if not slac_measurements.utils.wait_until(
                lambda: abs(self.my_wire.motor_rbv - position) < _WIRE_TOLERANCE,
            ):
                raise RuntimeError(
                    f"{self.my_wire.name} did not reach position {position} after 10s."
                )

        self.logger.info("Performing step scan mode")

        _initialize_step_with_retry()

        self.logger.info("Starting buffer acquisition for step scan...")
        acquisition_start = time.monotonic()
        acquisition_timeout_s = self._calculate_acquisition_timeout_s()
        self.my_buffer.start()

        positions = _get_step_positions()
        total_positions = len(positions)
        for index, position in enumerate(positions):
            _move_to_step_position(
                position=position,
                position_index=index,
                total_positions=total_positions,
                positions=positions,
            )

        self.logger.info("Retracting wire...")
        time.sleep(_WIRE_RETRACT_WAIT) # Wait for controller to stop moving
        self.my_wire.retract()
        time.sleep(_WIRE_RETRACT_WAIT) # Wait for wire to retract

        wire_position = self.my_wire.motor_rbv
        self.logger.info(
            "Wire retraction command issued. Motor position: %s",
            wire_position,
        )

        self.logger.info("Waiting for buffer acquisition to complete...")
        
        while not self.my_buffer.is_acquisition_complete():
            elapsed_s = time.monotonic() - acquisition_start
            if elapsed_s > acquisition_timeout_s:
                raise TimeoutError(
                    f"Timing buffer {self.my_buffer.number} did not complete after "
                    f"{elapsed_s:.1f}s (timeout={acquisition_timeout_s:.1f}s)."
                )
            time.sleep(0.1)

        self.logger.info(
            "Timing buffer %s acquisition complete after %.1f seconds",
            self.my_buffer.number,
            time.monotonic() - acquisition_start,
        )
   