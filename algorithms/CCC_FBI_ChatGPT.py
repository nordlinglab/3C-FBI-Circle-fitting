"""
3C-FBI: Combinatorial Convolution-based Circle Fitting in Blurry Images
======================================================================

Reference algorithm
-------------------
Combinatorial Convolution-based Circle Fitting in Blurry Images.

This module implements the algorithm described in the provided pseudocode:

1. Extract edgels from a binary edge image.
2. Randomly sample edgel triplets.
3. Fit a circle to each triplet in closed form.
4. Filter invalid circles.
5. Accumulate votes in a sparse integer vote map V[x, y, r].
6. Select the top-N voting cells.
7. Score each selected peak by summing votes inside a local cube.
8. Return the vote-weighted centroid of the best local cube.

Coordinate convention
---------------------
Images are NumPy arrays with shape (height, width). Pixel coordinates are
represented as (x, y), where:

    x = column index
    y = row index

The final circle is returned as:

    CircleEstimate(x, y, r)

where x and y are floating-point center coordinates and r is the radius.

Notes
-----
- The vote map is sparse: Python dict[(int, int, int), int].
- The top-N peak selection uses deterministic tie-breaking by first insertion
  order into the vote map, matching the pseudocode requirement.
- The sampling process is reproducible when random_state is provided.
- This implementation is intentionally explicit and easy to audit.

Example
-------
>>> import numpy as np
>>> from three_c_fbi import fit_circle_3cfbi
>>> edge = np.zeros((100, 100), dtype=np.uint8)
>>> # edge should contain nonzero pixels corresponding to detected edgels
>>> result = fit_circle_3cfbi(edge, r_min=5, r_max=30, random_state=0)
>>> print(result)

Author
------
Prepared as a standalone implementation for research use.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import argparse
import math
import os

import numpy as np

try:
    import cv2  # Optional: only required by the CLI image loader.
except ImportError:  # pragma: no cover
    cv2 = None


Point2D = Tuple[float, float]
VoteCell = Tuple[int, int, int]
SparseVoteMap = Dict[VoteCell, int]


@dataclass(frozen=True)
class CircleEstimate:
    """Final circle estimate.

    Attributes
    ----------
    x:
        Estimated x-coordinate of the circle center.
    y:
        Estimated y-coordinate of the circle center.
    r:
        Estimated circle radius.
    votes:
        Total local cube vote score W(p_i) for the selected peak.
    peak:
        Integer vote cell selected before local centroid refinement.
    n_edgels:
        Number of extracted edgels.
    n_triplets_requested:
        Number of random triplets requested.
    n_triplets_used:
        Number of sampled triplets actually processed.
    n_valid_votes:
        Number of accepted triplet circles after filtering.
    """

    x: float
    y: float
    r: float
    votes: int
    peak: VoteCell
    n_edgels: int
    n_triplets_requested: int
    n_triplets_used: int
    n_valid_votes: int


@dataclass(frozen=True)
class PeakScore:
    """Local cube score for one candidate peak."""

    peak: VoteCell
    weight: int
    centroid: Tuple[float, float, float]


class CircleFitError(RuntimeError):
    """Raised when the circle cannot be estimated from the given input."""


# -----------------------------------------------------------------------------
# Step 1: Edgel extraction
# -----------------------------------------------------------------------------


def extract_edgels(binary_edge_image: np.ndarray) -> np.ndarray:
    """Extract edgels from a binary edge image.

    Parameters
    ----------
    binary_edge_image:
        2D array. Nonzero pixels are treated as edgels.

    Returns
    -------
    np.ndarray
        Array of shape (n_edgels, 2), with rows [x, y].

    Raises
    ------
    ValueError
        If the input is not a 2D image.
    CircleFitError
        If fewer than three edgels are found.
    """

    image = np.asarray(binary_edge_image)
    if image.ndim != 2:
        raise ValueError(
            f"binary_edge_image must be a 2D array, got shape {image.shape}."
        )

    rows, cols = np.nonzero(image)
    edgels = np.column_stack((cols, rows)).astype(np.float64, copy=False)

    if edgels.shape[0] < 3:
        raise CircleFitError(
            f"At least 3 edgels are required; found {edgels.shape[0]}."
        )

    return edgels


# -----------------------------------------------------------------------------
# Step 2: Closed-form circle fitting from three points
# -----------------------------------------------------------------------------


def circle_from_triplet(
    p1: Sequence[float],
    p2: Sequence[float],
    p3: Sequence[float],
    *,
    determinant_eps: float = 0.0,
) -> Optional[Tuple[float, float, float]]:
    """Compute circle parameters from three 2D points in closed form.

    The formula uses the standard determinant solution for the unique circle
    through three non-collinear points.

    Parameters
    ----------
    p1, p2, p3:
        2D points represented as (x, y).
    determinant_eps:
        Collinearity threshold. The exact pseudocode uses D = 0. In floating
        point arithmetic, a small positive value can be useful. Default is 0.0
        to match the algorithm literally.

    Returns
    -------
    Optional[Tuple[float, float, float]]
        (x_center, y_center, radius), or None if the points are collinear or
        numerically degenerate.
    """

    x1, y1 = float(p1[0]), float(p1[1])
    x2, y2 = float(p2[0]), float(p2[1])
    x3, y3 = float(p3[0]), float(p3[1])

    # Determinant proportional to twice the signed triangle area.
    d = 2.0 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

    if abs(d) <= determinant_eps:
        return None

    q1 = x1 * x1 + y1 * y1
    q2 = x2 * x2 + y2 * y2
    q3 = x3 * x3 + y3 * y3

    cx = (q1 * (y2 - y3) + q2 * (y3 - y1) + q3 * (y1 - y2)) / d
    cy = (q1 * (x3 - x2) + q2 * (x1 - x3) + q3 * (x2 - x1)) / d

    radius_sq = (cx - x1) ** 2 + (cy - y1) ** 2
    if radius_sq <= 0.0 or not math.isfinite(radius_sq):
        return None

    r = math.sqrt(radius_sq)
    if not (math.isfinite(cx) and math.isfinite(cy) and math.isfinite(r)):
        return None

    return cx, cy, r


# -----------------------------------------------------------------------------
# Step 2: Triplet sampling and voting
# -----------------------------------------------------------------------------


def _make_rng(random_state: Optional[Union[int, np.random.Generator]]) -> np.random.Generator:
    if isinstance(random_state, np.random.Generator):
        return random_state
    return np.random.default_rng(random_state)


def _round_cell(x: float, y: float, r: float) -> VoteCell:
    """Round a circle hypothesis to the nearest integer vote cell.

    NumPy's rint uses banker's rounding for exact .5 cases. For continuous
    circle parameters this is rarely relevant, and the behavior is stable.
    """

    return int(np.rint(x)), int(np.rint(y)), int(np.rint(r))


def vote_random_triplets(
    edgels: np.ndarray,
    *,
    x_max: int,
    y_max: int,
    r_min: float,
    r_max: float,
    epsilon: float = 20.0,
    n_tri: int = 5000,
    random_state: Optional[Union[int, np.random.Generator]] = None,
    determinant_eps: float = 0.0,
) -> Tuple[SparseVoteMap, int, int]:
    """Sample random edgel triplets, fit circles, filter, and vote.

    Parameters
    ----------
    edgels:
        Array of shape (n_edgels, 2), with rows [x, y].
    x_max, y_max:
        Maximum valid image bounds. In a width x height image, these are usually
        width - 1 and height - 1.
    r_min, r_max:
        Accepted radius range.
    epsilon:
        Center tolerance outside the image bounds.
    n_tri:
        Number of triplets to sample.
    random_state:
        Seed or NumPy Generator for reproducibility.
    determinant_eps:
        Degeneracy threshold for the triplet determinant.

    Returns
    -------
    Tuple[SparseVoteMap, int, int]
        vote_map, n_triplets_used, n_valid_votes.
    """

    edgels = np.asarray(edgels, dtype=np.float64)
    if edgels.ndim != 2 or edgels.shape[1] != 2:
        raise ValueError("edgels must have shape (n_edgels, 2).")
    if edgels.shape[0] < 3:
        raise CircleFitError("At least 3 edgels are required to sample triplets.")
    if n_tri < 1:
        raise ValueError("n_tri must be >= 1.")
    if r_min > r_max:
        raise ValueError("r_min must be <= r_max.")

    rng = _make_rng(random_state)
    n_edgels = edgels.shape[0]

    vote_map: SparseVoteMap = {}
    n_valid_votes = 0

    for _ in range(n_tri):
        idx = rng.choice(n_edgels, size=3, replace=False)
        p1, p2, p3 = edgels[idx]

        circle = circle_from_triplet(
            p1,
            p2,
            p3,
            determinant_eps=determinant_eps,
        )
        if circle is None:
            continue

        x, y, r = circle

        # Filter 2: center outside bounds by more than epsilon.
        if x < -epsilon or y < -epsilon or x > x_max + epsilon or y > y_max + epsilon:
            continue

        # Filter 3: radius out of range.
        if r < r_min or r > r_max:
            continue

        cell = _round_cell(x, y, r)
        vote_map[cell] = vote_map.get(cell, 0) + 1
        n_valid_votes += 1

    return vote_map, n_tri, n_valid_votes


# -----------------------------------------------------------------------------
# Step 3: Top-N peak selection
# -----------------------------------------------------------------------------


def top_n_peaks(vote_map: Mapping[VoteCell, int], n_peaks: int = 5) -> List[VoteCell]:
    """Return the top-N vote cells with deterministic first-occurrence ties.

    Python dictionaries preserve insertion order. Sorting by -votes only is
    stable, so equal-vote cells retain first-insertion order.
    """

    if n_peaks < 1:
        raise ValueError("n_peaks must be >= 1.")
    if not vote_map:
        return []

    return [cell for cell, _count in sorted(vote_map.items(), key=lambda kv: -kv[1])[:n_peaks]]


# -----------------------------------------------------------------------------
# Step 4: Localized cube scoring
# -----------------------------------------------------------------------------


def score_peak_cube(
    peak: VoteCell,
    vote_map: Mapping[VoteCell, int],
    *,
    tau: int = 1,
) -> PeakScore:
    """Compute W(p_i) and the vote-weighted centroid inside a local cube.

    Parameters
    ----------
    peak:
        Integer candidate cell (x, y, r).
    vote_map:
        Sparse vote map V.
    tau:
        Cube half-width. tau=1 gives a 3x3x3 cube.

    Returns
    -------
    PeakScore
        Local weight and vote-weighted centroid.
    """

    if tau < 0:
        raise ValueError("tau must be >= 0.")

    px, py, pr = peak
    weight = 0
    sx = 0.0
    sy = 0.0
    sr = 0.0

    for dx in range(-tau, tau + 1):
        for dy in range(-tau, tau + 1):
            for dr in range(-tau, tau + 1):
                cell = (px + dx, py + dy, pr + dr)
                votes = int(vote_map.get(cell, 0))
                if votes == 0:
                    continue
                weight += votes
                sx += votes * cell[0]
                sy += votes * cell[1]
                sr += votes * cell[2]

    if weight == 0:
        # This should not happen for a peak selected from vote_map, but it makes
        # the function robust to arbitrary user-provided peaks.
        centroid = (float(px), float(py), float(pr))
    else:
        centroid = (sx / weight, sy / weight, sr / weight)

    return PeakScore(peak=peak, weight=weight, centroid=centroid)


def score_top_peaks(
    peaks: Sequence[VoteCell],
    vote_map: Mapping[VoteCell, int],
    *,
    tau: int = 1,
) -> List[PeakScore]:
    """Score all selected peaks using local cube accumulation."""

    return [score_peak_cube(peak, vote_map, tau=tau) for peak in peaks]


# -----------------------------------------------------------------------------
# Step 5: Final estimate
# -----------------------------------------------------------------------------


def select_best_peak(scores: Sequence[PeakScore]) -> PeakScore:
    """Select the peak with maximum W(p_i), breaking ties by first occurrence."""

    if not scores:
        raise CircleFitError("No peak scores are available for final selection.")

    # max is stable with respect to first occurrence for equal keys.
    return max(scores, key=lambda score: score.weight)


def fit_circle_3cfbi(
    binary_edge_image: np.ndarray,
    *,
    x_max: Optional[int] = None,
    y_max: Optional[int] = None,
    r_min: float,
    r_max: float,
    epsilon: float = 20.0,
    n_tri: int = 5000,
    n_peaks: int = 5,
    tau: int = 1,
    random_state: Optional[Union[int, np.random.Generator]] = None,
    determinant_eps: float = 0.0,
    return_debug: bool = False,
) -> Union[CircleEstimate, Tuple[CircleEstimate, dict]]:
    """Estimate a circle from a binary edge image using 3C-FBI.

    Parameters
    ----------
    binary_edge_image:
        2D binary or edge image. Nonzero pixels are edgels.
    x_max, y_max:
        Valid image bounds. If omitted, x_max = width - 1 and y_max = height - 1.
    r_min, r_max:
        Accepted radius range.
    epsilon:
        Center tolerance in pixels.
    n_tri:
        Number of random triplets.
    n_peaks:
        Number of top voting cells to score.
    tau:
        Half-width of the local scoring cube.
    random_state:
        Seed or NumPy Generator for reproducible sampling.
    determinant_eps:
        Numerical threshold for discarding nearly collinear triplets.
    return_debug:
        If True, return (estimate, debug_dict).

    Returns
    -------
    CircleEstimate or Tuple[CircleEstimate, dict]
        Final refined circle estimate, optionally with debugging internals.
    """

    image = np.asarray(binary_edge_image)
    if image.ndim != 2:
        raise ValueError(f"binary_edge_image must be 2D, got shape {image.shape}.")

    height, width = image.shape
    if x_max is None:
        x_max = width - 1
    if y_max is None:
        y_max = height - 1

    edgels = extract_edgels(image)

    vote_map, n_triplets_used, n_valid_votes = vote_random_triplets(
        edgels,
        x_max=int(x_max),
        y_max=int(y_max),
        r_min=float(r_min),
        r_max=float(r_max),
        epsilon=float(epsilon),
        n_tri=int(n_tri),
        random_state=random_state,
        determinant_eps=float(determinant_eps),
    )

    if not vote_map:
        raise CircleFitError(
            "No valid circle hypotheses survived filtering. "
            "Consider checking the edge image, radius range, epsilon, or n_tri."
        )

    peaks = top_n_peaks(vote_map, n_peaks=n_peaks)
    scores = score_top_peaks(peaks, vote_map, tau=tau)
    best = select_best_peak(scores)

    x, y, r = best.centroid
    estimate = CircleEstimate(
        x=float(x),
        y=float(y),
        r=float(r),
        votes=int(best.weight),
        peak=best.peak,
        n_edgels=int(edgels.shape[0]),
        n_triplets_requested=int(n_tri),
        n_triplets_used=int(n_triplets_used),
        n_valid_votes=int(n_valid_votes),
    )

    if return_debug:
        debug = {
            "edgels": edgels,
            "vote_map": vote_map,
            "peaks": peaks,
            "scores": scores,
        }
        return estimate, debug

    return estimate


# -----------------------------------------------------------------------------
# Convenience utilities
# -----------------------------------------------------------------------------


def draw_circle_edge_image(
    shape: Tuple[int, int],
    x: float,
    y: float,
    r: float,
    *,
    thickness: int = 1,
) -> np.ndarray:
    """Create a synthetic binary circle edge image for testing/demo purposes."""

    if cv2 is None:
        raise ImportError("OpenCV is required for draw_circle_edge_image.")

    image = np.zeros(shape, dtype=np.uint8)
    cv2.circle(
        image,
        center=(int(round(x)), int(round(y))),
        radius=int(round(r)),
        color=255,
        thickness=int(thickness),
        lineType=cv2.LINE_8,
    )
    return image


def load_binary_image(path: str, *, threshold: int = 0) -> np.ndarray:
    """Load an image as a binary edge image.

    Pixels greater than threshold become 255; all others become 0.
    """

    if cv2 is None:
        raise ImportError("OpenCV is required for image loading. Install opencv-python.")

    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")

    return np.where(image > threshold, 255, 0).astype(np.uint8)


def _format_estimate(estimate: CircleEstimate) -> str:
    return (
        f"x={estimate.x:.6f}, y={estimate.y:.6f}, r={estimate.r:.6f}, "
        f"local_votes={estimate.votes}, peak={estimate.peak}, "
        f"edgels={estimate.n_edgels}, valid_votes={estimate.n_valid_votes}/"
        f"{estimate.n_triplets_requested}"
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Command-line interface.

    Example
    -------
    python three_c_fbi.py path/to/edge.png --r-min 8 --r-max 20 --seed 0
    """

    parser = argparse.ArgumentParser(
        description="3C-FBI circle fitting from a binary edge image."
    )
    parser.add_argument("image", help="Path to a binary/edge image.")
    parser.add_argument("--r-min", type=float, required=True, help="Minimum radius.")
    parser.add_argument("--r-max", type=float, required=True, help="Maximum radius.")
    parser.add_argument("--epsilon", type=float, default=20.0, help="Center tolerance in px.")
    parser.add_argument("--n-tri", type=int, default=5000, help="Number of triplets.")
    parser.add_argument("--n-peaks", type=int, default=5, help="Number of top peaks.")
    parser.add_argument("--tau", type=int, default=1, help="Cube half-width.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument(
        "--threshold",
        type=int,
        default=0,
        help="Input binarization threshold. Pixels > threshold are edgels.",
    )
    parser.add_argument(
        "--determinant-eps",
        type=float,
        default=0.0,
        help="Threshold for rejecting nearly collinear triplets.",
    )

    args = parser.parse_args(argv)

    if not os.path.exists(args.image):
        raise FileNotFoundError(args.image)

    binary = load_binary_image(args.image, threshold=args.threshold)
    estimate = fit_circle_3cfbi(
        binary,
        r_min=args.r_min,
        r_max=args.r_max,
        epsilon=args.epsilon,
        n_tri=args.n_tri,
        n_peaks=args.n_peaks,
        tau=args.tau,
        random_state=args.seed,
        determinant_eps=args.determinant_eps,
    )

    print(_format_estimate(estimate))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
