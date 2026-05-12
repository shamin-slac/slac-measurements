import sys
from unittest import TestCase
from unittest.mock import MagicMock, patch
import numpy as np

if "edef" not in sys.modules:
    sys.modules["edef"] = MagicMock()

from slac_devices.wire import Wire
from slac_measurements.tmit_loss import TMITLoss


def _make_mock_bpm(name, z_location):
    """Create a mock BPM with a name and z_location."""
    bpm = MagicMock()
    bpm.z_location = z_location
    bpm.name = name
    return bpm


def _make_mock_beampath(bpm_dict):
    """Create a mock Beampath with a .bpms property returning bpm_dict."""
    beampath = MagicMock()
    beampath.bpms = bpm_dict
    return beampath


def _make_wire(bpms_before, bpms_after):
    """Create a Wire instance with metadata listing upstream/downstream BPM names."""
    tmitloss = MagicMock()
    tmitloss.upstream = bpms_before
    tmitloss.downstream = bpms_after
    metadata = MagicMock()
    metadata.tmitloss = tmitloss
    wire = Wire.model_construct(metadata=metadata)
    return wire


class TestCalcTmitLoss(TestCase):
    """Test the TMIT loss calculation math in isolation."""

    def _make_instance(self, idx_before, idx_after):
        """Create a TMITLoss instance with given indices, bypassing validation."""
        return TMITLoss.model_construct(
            idx_before=idx_before,
            idx_after=idx_after,
        )

    def test_no_loss_when_all_bpms_identical(self):
        instance = self._make_instance([0, 1], [2, 3])
        data = np.array(
            [
                [2.0, 4.0, 6.0],
                [2.0, 4.0, 6.0],
                [2.0, 4.0, 6.0],
                [2.0, 4.0, 6.0],
            ]
        )

        result = instance.calc_tmit_loss(data)

        np.testing.assert_allclose(result, [0.0, 0.0, 0.0], atol=1e-10)

    def test_known_loss_values(self):
        instance = self._make_instance([0, 1], [2, 3])
        data = np.array(
            [
                [2.0, 4.0],
                [2.0, 4.0],
                [1.0, 4.0],
                [1.0, 4.0],
            ]
        )

        result = instance.calc_tmit_loss(data)

        # Hand-computed:
        # row_medians = [3, 3, 2.5, 2.5]
        # ironed = [[2/3, 4/3], [2/3, 4/3], [0.4, 1.6], [0.4, 1.6]]
        # mean_iron_before = [2/3, 4/3]
        # normed = [[1, 1], [1, 1], [0.6, 1.2], [0.6, 1.2]]
        # mean_before = [1, 1], mean_after = [0.6, 1.2]
        # loss = [40, -20]
        np.testing.assert_allclose(result, [40.0, -20.0], atol=1e-10)

    def test_uniform_fractional_loss_gives_zero(self):
        """A constant fractional loss across all shots normalizes away."""
        instance = self._make_instance([0, 1], [2, 3])
        data = np.array(
            [
                [100.0, 200.0, 150.0],
                [100.0, 200.0, 150.0],
                [50.0, 100.0, 75.0],
                [50.0, 100.0, 75.0],
            ]
        )

        result = instance.calc_tmit_loss(data)

        np.testing.assert_allclose(result, [0.0, 0.0, 0.0], atol=1e-10)


class TestRunSetup(TestCase):
    """Test that model validation wires up BPMs and indices correctly."""

    @patch("slac_measurements.tmit_loss.create_beampath")
    def test_bpms_sorted_by_z_location(self, mock_create_beampath):
        bpm_a = _make_mock_bpm("BPM_A", z_location=10.0)
        bpm_b = _make_mock_bpm("BPM_B", z_location=5.0)
        bpm_c = _make_mock_bpm("BPM_C", z_location=20.0)
        mock_create_beampath.return_value = _make_mock_beampath(
            {"BPM_A": bpm_a, "BPM_B": bpm_b, "BPM_C": bpm_c}
        )

        wire = _make_wire(
            bpms_before=["BPM_B"],
            bpms_after=["BPM_C"],
        )

        instance = TMITLoss(
            buffer=MagicMock(),
            beampath="TEST",
            beam_profile_device=wire,
        )

        self.assertEqual(list(instance.bpms.keys()), ["BPM_B", "BPM_A", "BPM_C"])

    @patch("slac_measurements.tmit_loss.create_beampath")
    def test_indices_resolve_correctly(self, mock_create_beampath):
        bpm_a = _make_mock_bpm("BPM_A", z_location=1.0)
        bpm_b = _make_mock_bpm("BPM_B", z_location=2.0)
        bpm_c = _make_mock_bpm("BPM_C", z_location=3.0)
        bpm_d = _make_mock_bpm("BPM_D", z_location=4.0)
        mock_create_beampath.return_value = _make_mock_beampath(
            {"BPM_A": bpm_a, "BPM_B": bpm_b, "BPM_C": bpm_c, "BPM_D": bpm_d}
        )

        wire = _make_wire(
            bpms_before=["BPM_A", "BPM_B"],
            bpms_after=["BPM_C", "BPM_D"],
        )

        instance = TMITLoss(
            buffer=MagicMock(),
            beampath="TEST",
            beam_profile_device=wire,
        )

        self.assertEqual(instance.idx_before, [0, 1])
        self.assertEqual(instance.idx_after, [2, 3])

    @patch("slac_measurements.tmit_loss.create_beampath")
    def test_missing_bpms_skipped(self, mock_create_beampath):
        bpm_a = _make_mock_bpm("BPM_A", z_location=1.0)
        bpm_b = _make_mock_bpm("BPM_B", z_location=2.0)
        mock_create_beampath.return_value = _make_mock_beampath(
            {"BPM_A": bpm_a, "BPM_B": bpm_b}
        )

        wire = _make_wire(
            bpms_before=["BPM_A", "BPM_MISSING"],
            bpms_after=["BPM_B", "BPM_GONE"],
        )

        instance = TMITLoss(
            buffer=MagicMock(),
            beampath="TEST",
            beam_profile_device=wire,
        )

        self.assertEqual(instance.idx_before, [0])
        self.assertEqual(instance.idx_after, [1])

    @patch("slac_measurements.tmit_loss.create_beampath")
    def test_empty_beampath_raises(self, mock_create_beampath):
        mock_create_beampath.return_value = _make_mock_beampath({})

        wire = _make_wire(bpms_before=[], bpms_after=[])

        with self.assertRaises(LookupError):
            TMITLoss(
                buffer=MagicMock(),
                beampath="TEST",
                beam_profile_device=wire,
            )


class TestMeasure(TestCase):
    """Test the full measure pipeline with mocked data collection."""

    @patch("slac_measurements.tmit_loss.collect_with_size_check")
    @patch("slac_measurements.tmit_loss.create_beampath")
    def test_measure_returns_numpy_array(self, mock_create_beampath, mock_collect):
        bpm_a = _make_mock_bpm("BPM_A", z_location=1.0)
        bpm_b = _make_mock_bpm("BPM_B", z_location=2.0)
        bpm_c = _make_mock_bpm("BPM_C", z_location=3.0)
        bpm_d = _make_mock_bpm("BPM_D", z_location=4.0)
        mock_create_beampath.return_value = _make_mock_beampath(
            {"BPM_A": bpm_a, "BPM_B": bpm_b, "BPM_C": bpm_c, "BPM_D": bpm_d}
        )

        wire = _make_wire(
            bpms_before=["BPM_A", "BPM_B"],
            bpms_after=["BPM_C", "BPM_D"],
        )

        mock_collect.side_effect = [
            np.array([2.0, 4.0]),
            np.array([2.0, 4.0]),
            np.array([2.0, 4.0]),
            np.array([2.0, 4.0]),
        ]

        instance = TMITLoss(
            buffer=MagicMock(),
            beampath="TEST",
            beam_profile_device=wire,
        )

        result = instance.measure()

        self.assertIsInstance(result, np.ndarray)
        np.testing.assert_allclose(result, [0.0, 0.0], atol=1e-10)

    @patch("slac_measurements.tmit_loss.collect_with_size_check")
    @patch("slac_measurements.tmit_loss.create_beampath")
    def test_measure_with_loss(self, mock_create_beampath, mock_collect):
        bpm_a = _make_mock_bpm("BPM_A", z_location=1.0)
        bpm_b = _make_mock_bpm("BPM_B", z_location=2.0)
        bpm_c = _make_mock_bpm("BPM_C", z_location=3.0)
        bpm_d = _make_mock_bpm("BPM_D", z_location=4.0)
        mock_create_beampath.return_value = _make_mock_beampath(
            {"BPM_A": bpm_a, "BPM_B": bpm_b, "BPM_C": bpm_c, "BPM_D": bpm_d}
        )

        wire = _make_wire(
            bpms_before=["BPM_A", "BPM_B"],
            bpms_after=["BPM_C", "BPM_D"],
        )

        mock_collect.side_effect = [
            np.array([2.0, 4.0]),
            np.array([2.0, 4.0]),
            np.array([1.0, 4.0]),
            np.array([1.0, 4.0]),
        ]

        instance = TMITLoss(
            buffer=MagicMock(),
            beampath="TEST",
            beam_profile_device=wire,
        )

        result = instance.measure()

        np.testing.assert_allclose(result, [40.0, -20.0], atol=1e-10)
