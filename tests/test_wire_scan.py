from unittest import TestCase
from unittest.mock import MagicMock, patch

from slac_measurements.wires.scan import WireBeamProfileMeasurement


class WireBeamProfileMeasurementTest(TestCase):
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
            beam_profile_device=MagicMock(),
            beampath="TEST",
            fitting_method="super_gaussian",
        )

        result = measurement.measure()

        self.assertEqual(result, "analysis-result")
        self.assertEqual(measurement.collection_result, "collection-result")
        mock_collection_cls.assert_called_once_with(
            beam_profile_device=measurement.beam_profile_device,
            beampath="TEST",
        )
        mock_collection.measure.assert_called_once_with(scan_type="step")
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
            beam_profile_device=MagicMock(),
            beampath="TEST",
            fitting_method="gaussian",
        )

        result = measurement.measure(fitting_method="asymmetric_gaussian")

        self.assertEqual(result, "analysis-result")
        mock_collection.measure.assert_called_once_with(scan_type="step")
        mock_analysis_cls.assert_called_once_with(
            collection_result="collection-result",
            fitting_method="asymmetric_gaussian",
        )

    @patch("slac_measurements.wires.scan.WireMeasurementAnalysis")
    def test_analyze_can_refit_with_override(self, mock_analysis_cls):
        mock_analysis = mock_analysis_cls.return_value
        mock_analysis.analyze.return_value = "analysis-result"

        measurement = WireBeamProfileMeasurement(
            beam_profile_device=MagicMock(),
            beampath="TEST",
            fitting_method="gaussian",
        )
        measurement.collection_result = "collection-result"

        result = measurement.analyze(fitting_method="super_gaussian")

        self.assertEqual(result, "analysis-result")
        mock_analysis_cls.assert_called_once_with(
            collection_result="collection-result",
            fitting_method="super_gaussian",
        )

    def test_analyze_raises_if_no_collection_result(self):
        measurement = WireBeamProfileMeasurement(
            beam_profile_device=MagicMock(),
            beampath="TEST",
        )

        with self.assertRaises(RuntimeError):
            measurement.analyze()
