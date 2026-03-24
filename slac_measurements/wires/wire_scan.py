from slac_devices.wire import Wire
import slac_measurements.beam_profile
from slac_measurements.wires.ws_collection import WireMeasurementCollection
from slac_measurements.wires.ws_analysis import WireMeasurementAnalysis
from slac_measurements.wires.ws_analysis_results import WireMeasurementAnalysisResult


class WireBeamProfileMeasurement(slac_measurements.beam_profile.BeamProfileMeasurement):
    """
    Orchestrates a full wire scan: accepts a wire device and beampath,
    instantiates a WireMeasurementCollection, runs the scan, and returns
    the analyzed result.

    Attributes:
        beam_profile_device (Wire): Wire device for the scan.
        beampath (str): Beamline identifier passed to the collection.
    """

    name: str = "Wire Beam Profile Measurement"
    beam_profile_device: Wire
    beampath: str

    def measure(self, scan_type: str = "step") -> WireMeasurementAnalysisResult:
        """
        Instantiate a WireMeasurementCollection, run the scan, analyze, and
        return the result.

        Parameters
        ----------
        scan_type : str, optional
            ``"on_the_fly"`` or ``"step"`` (default).

        Returns
        -------
        WireMeasurementAnalysisResult
            Fit results, RMS beam sizes, and organized profile data.
        """
        collection = WireMeasurementCollection(
            beam_profile_device=self.beam_profile_device,
            beampath=self.beampath,
        )
        collection_result = collection.measure(scan_type=scan_type)
        analysis = WireMeasurementAnalysis(collection_result=collection_result)
        return analysis.analyze()
