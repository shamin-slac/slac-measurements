from unittest import TestCase
from unittest.mock import MagicMock, patch
from datetime import datetime
import numpy as np

from slac_devices.wire import Wire
from slac_measurements.wires.scan import WireBeamProfileMeasurement
from slac_measurements.wires.collection_results import (
    MeasurementMetadata,
    WireMeasurementCollectionResult,
)


class WireBeamProfileMeasurementTest(TestCase):
    @staticmethod
    def _make_wire_device() -> Wire:
        # Create an instance without invoking device initialization side effects.
        return Wire.__new__(Wire)

    @staticmethod
    def _make_collection_result() -> WireMeasurementCollectionResult:
        metadata = MeasurementMetadata(
            wire_name="WIRE",
            area="TEST",
            beampath="TEST",
            detectors=["D1"],
            default_detector="D1",
            scan_ranges={"x": (0, 1)},
            timestamp=datetime.now(),
            active_profiles=["x"],
            install_angle=45.0,
        )
        return WireMeasurementCollectionResult(
            raw_data={"WIRE": np.array([0.0, 1.0])},
            metadata=metadata,
        )

    @patch("slac_measurements.wires.scan.WireMeasurementAnalysis")
    @patch("slac_measurements.wires.scan.WireMeasurementCollection")
    def test_measure_uses_configured_fitting_method(
        self,
        mock_collection_cls,
        mock_analysis_cls,
    ):
        mock_collection = mock_collection_cls.return_value
        mock_collection.measure.return_value = "collection-result"

        mock_analysis = mock_analysis_cls.return_value
        mock_analysis.analyze.return_value = "analysis-result"

        measurement = WireBeamProfileMeasurement(
            beam_profile_device=self._make_wire_device(),
            beampath="TEST",
        )

        result = measurement.measure(fitting_method="super_gaussian")

        self.assertEqual(result, "analysis-result")
        self.assertEqual(measurement.collection_result, "collection-result")
        mock_collection_cls.assert_called_once_with(
            beam_profile_device=measurement.beam_profile_device,
            beampath="TEST",
        )
        mock_collection.measure.assert_called_once_with(scan_mode="step")
        mock_analysis_cls.assert_called_once_with(
            collection_result="collection-result",
            fitting_method="super_gaussian",
        )

    @patch("slac_measurements.wires.scan.WireMeasurementAnalysis")
    @patch("slac_measurements.wires.scan.WireMeasurementCollection")
    def test_measure_allows_fitting_method_override(
        self,
        mock_collection_cls,
        mock_analysis_cls,
    ):
        mock_collection = mock_collection_cls.return_value
        mock_collection.measure.return_value = "collection-result"

        mock_analysis = mock_analysis_cls.return_value
        mock_analysis.analyze.return_value = "analysis-result"

        measurement = WireBeamProfileMeasurement(
            beam_profile_device=self._make_wire_device(),
            beampath="TEST",
        )

        result = measurement.measure(fitting_method="asymmetric_gaussian")

        self.assertEqual(result, "analysis-result")
        mock_collection.measure.assert_called_once_with(scan_mode="step")
        mock_analysis_cls.assert_called_once_with(
            collection_result="collection-result",
            fitting_method="asymmetric_gaussian",
        )

    @patch("slac_measurements.wires.scan.WireMeasurementAnalysis")
    def test_analyze_can_refit_with_override(self, mock_analysis_cls):
        mock_analysis = mock_analysis_cls.return_value
        mock_analysis.analyze.return_value = "analysis-result"

        measurement = WireBeamProfileMeasurement(
            beam_profile_device=self._make_wire_device(),
            beampath="TEST",
        )
        measurement.collection_result = self._make_collection_result()

        result = measurement.analyze(fitting_method="super_gaussian")

        self.assertEqual(result, "analysis-result")
        mock_analysis_cls.assert_called_once_with(
            collection_result=measurement.collection_result,
            fitting_method="super_gaussian",
        )
        mock_analysis.analyze.assert_called_once_with(rms_detector=None)

    # measure() constructs WireMeasurementCollection internally, so we patch
    # it to keep this unit test isolated from collection setup behavior.
    @patch("slac_measurements.wires.scan.WireMeasurementCollection")
    def test_measure_passes_rms_detector_override(self, mock_collection_cls):
        mock_collection_cls.return_value.measure.return_value = "collection-result"

        measurement = WireBeamProfileMeasurement(
            beam_profile_device=self._make_wire_device(),
            beampath="TEST",
        )

        with patch.object(
            WireBeamProfileMeasurement,
            "analyze",
            autospec=True,
            return_value="analysis-result",
        ) as mock_analyze:
            measurement.measure(
                fitting_method="gaussian",
                rms_detector="D2",
            )

        mock_analyze.assert_called_once_with(
            measurement,
            fitting_method="gaussian",
            rms_detector="D2",
        )

    def test_analyze_raises_if_no_collection_result(self):
        measurement = WireBeamProfileMeasurement(
            beam_profile_device=self._make_wire_device(),
            beampath="TEST",
        )

        with self.assertRaises(RuntimeError):
            measurement.analyze(fitting_method="gaussian")
