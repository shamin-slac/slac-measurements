import numpy as np

from slac_measurements.wires.collection_results import WireMeasurementCollectionResult
from slac_measurements.wires.coordinates import xy_to_stage


def correct_jitter(
    collection_result: WireMeasurementCollectionResult,
    beampath: str,
    physics_model: str = "BLEM",
    include_energy: bool = True,
) -> WireMeasurementCollectionResult:
    """Apply jitter correction to a wire scan collection result.

    Uses BPM position data and transport matrices to reconstruct beam
    position jitter at the wire location, then subtracts it from the
    wire position data.

    Parameters
    ----------
    collection_result : WireMeasurementCollectionResult
        Raw collection result containing wire positions and BPM x/y buffer data.
    beampath : str
        Beam path identifier for the model (e.g., "SC_HXR", "SC_DIAG0").
    physics_model : str
        Model source for R-matrix retrieval. Default "BLEM".
    include_energy : bool
        Whether to include the dispersion (energy) term in the orbit fit.

    Returns
    -------
    WireMeasurementCollectionResult
        New collection result with corrected wire positions,
        jitter_corrected=True, and jitter_rms_x/jitter_rms_y populated.

    Raises
    ------
    ValueError
        If no BPM data is found in the collection result or if fewer than
        2 BPMs have valid data.
    """
    wire_name = collection_result.metadata.wire_name
    install_angle = collection_result.metadata.install_angle
    raw_data = collection_result.raw_data

    bpm_x_data, bpm_y_data, bpm_names = _extract_bpm_data(raw_data)

    if len(bpm_names) < 2:
        raise ValueError(
            f"Jitter correction requires at least 2 BPMs with valid data, "
            f"found {len(bpm_names)}."
        )

    rmat_x, rmat_y = get_jitter_rmat(
        wire_name, bpm_names, beampath, physics_model, include_energy
    )

    wire_positions = raw_data[wire_name].copy()

    jitter_x, jitter_y = _compute_orbit_fit(bpm_x_data, bpm_y_data, rmat_x, rmat_y)

    jitter_stage = xy_to_stage(jitter_x, jitter_y, install_angle)
    corrected_positions = wire_positions - jitter_stage

    corrected_raw_data = dict(raw_data)
    corrected_raw_data[wire_name] = corrected_positions

    return WireMeasurementCollectionResult(
        raw_data=corrected_raw_data,
        metadata=collection_result.metadata,
        jitter_corrected=True,
        jitter_rms_x=float(np.std(jitter_x)),
        jitter_rms_y=float(np.std(jitter_y)),
    )


def get_jitter_rmat(
    wire_name: str,
    bpm_names: list[str],
    beampath: str,
    physics_model: str = "BLEM",
    include_energy: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Get R-matrices from wire to each BPM for orbit fitting.

    Parameters
    ----------
    wire_name : str
        Wire device name (MAD element name).
    bpm_names : list[str]
        BPM control names (e.g., ["BPMS:HTR:120", "BPMS:HTR:320"]).
    beampath : str
        Beam path identifier for the model.
    physics_model : str
        Model source. Default "BLEM".
    include_energy : bool
        If True, include R16/R36 dispersion term (3-column matrix).
        If False, use only R11,R12 / R33,R34 (2-column matrix).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (rmat_x, rmat_y) where:
        - rmat_x is [N_bpms x 2] (or [N_bpms x 3] with energy)
        - rmat_y is [N_bpms x 2] (or [N_bpms x 3] with energy)
        Each row relates BPM position reading to beam state at wire.
    """
    from meme.model import Model

    model = Model(beampath, model_source=physics_model, use_design=False)

    n_cols = 3 if include_energy else 2
    rmat_x = np.zeros((len(bpm_names), n_cols))
    rmat_y = np.zeros((len(bpm_names), n_cols))

    for i, bpm_name in enumerate(bpm_names):
        rmat_6x6 = model.get_rmat(from_device=wire_name, to_device=bpm_name)

        # x plane: position at BPM from (x, x') at wire
        rmat_x[i, 0] = rmat_6x6[0, 0]  # R11
        rmat_x[i, 1] = rmat_6x6[0, 1]  # R12
        if include_energy:
            rmat_x[i, 2] = rmat_6x6[0, 5]  # R16

        # y plane: position at BPM from (y, y') at wire
        rmat_y[i, 0] = rmat_6x6[2, 2]  # R33
        rmat_y[i, 1] = rmat_6x6[2, 3]  # R34
        if include_energy:
            rmat_y[i, 2] = rmat_6x6[2, 5]  # R36

    return rmat_x, rmat_y


def _extract_bpm_data(
    raw_data: dict,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Extract BPM x/y position data from raw_data dictionary.

    BPM entries are keyed by device name (starting with 'BPM') and contain
    a dict with 'x' and 'y' buffer arrays.

    Returns
    -------
    tuple
        (bpm_x_array, bpm_y_array, bpm_names) where arrays are
        [N_bpms x N_pulses] and bpm_names are device names.
    """
    bpm_keys = sorted(k for k in raw_data if k.startswith("BPM"))

    bpm_names = []
    x_arrays = []
    y_arrays = []

    for name in bpm_keys:
        bpm_data = raw_data[name]

        if not isinstance(bpm_data, dict):
            continue

        x_arr = bpm_data.get("x")
        y_arr = bpm_data.get("y")

        if x_arr is None or y_arr is None:
            continue
        if np.all(np.isnan(x_arr)) or np.all(np.isnan(y_arr)):
            continue

        bpm_names.append(name)
        x_arrays.append(x_arr)
        y_arrays.append(y_arr)

    if not bpm_names:
        raise ValueError("No valid BPM position data found in collection result.")

    bpm_x_data = np.array(x_arrays)  # [N_bpms x N_pulses]
    bpm_y_data = np.array(y_arrays)  # [N_bpms x N_pulses]

    return bpm_x_data, bpm_y_data, bpm_names


def _compute_orbit_fit(
    bpm_x_data: np.ndarray,
    bpm_y_data: np.ndarray,
    rmat_x: np.ndarray,
    rmat_y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute beam position at wire from BPM readings via least-squares orbit fit.

    Parameters
    ----------
    bpm_x_data : np.ndarray
        BPM x positions [N_bpms x N_pulses] in mm.
    bpm_y_data : np.ndarray
        BPM y positions [N_bpms x N_pulses] in mm.
    rmat_x : np.ndarray
        X-plane transport matrix [N_bpms x 2 or 3].
    rmat_y : np.ndarray
        Y-plane transport matrix [N_bpms x 2 or 3].

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (jitter_x, jitter_y) - reconstructed beam position jitter at wire
        for each pulse, in um (matching wire position units).
    """
    # Compute deviations from mean (mm)
    delta_x = bpm_x_data - bpm_x_data.mean(axis=1, keepdims=True)
    delta_y = bpm_y_data - bpm_y_data.mean(axis=1, keepdims=True)

    # Convert mm -> m for orbit fit
    delta_x_m = delta_x * 1e-3
    delta_y_m = delta_y * 1e-3

    # Least-squares fit: rmat @ orbit = delta_bpm for each pulse
    # orbit_x shape: [n_params x N_pulses], orbit_x[0] is x position at wire
    orbit_x, _, _, _ = np.linalg.lstsq(rmat_x, delta_x_m, rcond=None)
    orbit_y, _, _, _ = np.linalg.lstsq(rmat_y, delta_y_m, rcond=None)

    # Position at wire (first element of orbit vector), convert m -> um
    jitter_x_um = orbit_x[0, :] * 1e6
    jitter_y_um = orbit_y[0, :] * 1e6

    return jitter_x_um, jitter_y_um
