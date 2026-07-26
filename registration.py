"""
registration.py — Robust star-based image registration
========================================================

Two-stage registration for astronomical images:

  Stage 1: Star-based triangle matching (primary)
    - Detect stars via background estimation + peak finding
    - Build rotation/scale-invariant triangle descriptors
    - Hash-match triangles between reference and source
    - RANSAC-fit a similarity transform from matched stars

  Stage 2: Phase correlation (fallback)
    - Used only when star detection or matching fails
    - Median-subtracted + Hann-windowed cross-power spectrum

Both stages support GPU acceleration via CuPy when available.

Public API
----------
    register_frames(frames, reference_idx=0, ...)
        -> list of 2x3 float64 ndarray (affine matrices)
"""

import numpy as np
from itertools import combinations

try:
    import cupy as cp
    _CUPY_AVAILABLE = True
except ImportError:
    _CUPY_AVAILABLE = False

from scipy.ndimage import maximum_filter


# ---------------------------------------------------------------------------
#  Configuration defaults
# ---------------------------------------------------------------------------

MAX_ROTATION_DEG = 5.0
MIN_PEAK_HEIGHT  = 5.0     # peak-to-mean ratio for phase correlation fallback
MAX_STARS        = 200      # max stars to use for matching
STAR_DETECTION_SIGMA = 5.0 # detection threshold in background sigma units
RANSAC_REPROJ_THRESHOLD = 2.0  # pixels


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _to_luminance(frame):
    """Return (H, W) float32 luminance from (H, W) or (H, W, C)."""
    if frame.ndim == 3:
        return (0.2126 * frame[:, :, 0] +
                0.7152 * frame[:, :, 1] +
                0.0722 * frame[:, :, 2]).astype(np.float32)
    return frame.astype(np.float32)


def _hann2d(H, W):
    """2-D Hann window, shape (H, W), float32."""
    return np.outer(np.hanning(H).astype(np.float32),
                    np.hanning(W).astype(np.float32))


# ---------------------------------------------------------------------------
#  Star detection
# ---------------------------------------------------------------------------

def _star_bandpass(img, star_fwhm=4.0, use_gpu=False):
    """
    Isolate point sources from extended nebulosity using a bandpass filter.

    Subtracts a smoothed version of the image (kills nebula) from a
    lightly-smoothed version (preserves stars).  The result contains only
    compact features at the stellar scale.

    GPU acceleration: the two gaussian_filter calls are the expensive
    part.  When use_gpu=True, they run on CuPy and the result is
    transferred back to CPU as float64.  This ensures all subsequent
    peak detection and centroid logic is identical regardless of path.

    Parameters
    ----------
    img : (H, W) float64 numpy array
    star_fwhm : float
    use_gpu : bool

    Returns
    -------
    bandpass : (H, W) float64 numpy array
    """
    sigma_star = star_fwhm / 2.355

    if use_gpu and _CUPY_AVAILABLE:
        from cupyx.scipy.ndimage import gaussian_filter as cu_gaussian_filter
        img_gpu = cp.asarray(img)
        small = cu_gaussian_filter(img_gpu, sigma=sigma_star)
        large = cu_gaussian_filter(img_gpu, sigma=sigma_star * 5.0)
        bandpass = cp.asnumpy(small - large)
        del img_gpu, small, large
    else:
        from scipy.ndimage import gaussian_filter
        small = gaussian_filter(img, sigma=sigma_star)
        large = gaussian_filter(img, sigma=sigma_star * 5.0)
        bandpass = small - large

    return bandpass


def _measure_sharpness(img, y0, x0, half=5):
    """
    Measure how compact/star-like a peak is.

    Returns the ratio of flux in the inner region (r < half/2) to total
    flux in the box (r < half).  Stars concentrate flux in the centre;
    nebula knots spread it out.

    A real star typically has sharpness > 0.4; nebula features < 0.3.
    """
    H, W = img.shape
    y_lo, y_hi = max(0, y0 - half), min(H, y0 + half + 1)
    x_lo, x_hi = max(0, x0 - half), min(W, x0 + half + 1)
    patch = img[y_lo:y_hi, x_lo:x_hi]
    patch = np.maximum(patch, 0)

    total = patch.sum()
    if total <= 0:
        return 0.0

    cy, cx = y0 - y_lo, x0 - x_lo
    yy, xx = np.mgrid[0:patch.shape[0], 0:patch.shape[1]]
    r2 = (yy - cy)**2 + (xx - cx)**2
    inner_half = half / 2.0
    inner_flux = patch[r2 <= inner_half**2].sum()

    return float(inner_flux / total)


def _gaussian_centroid(img, y0, x0, half):
    """
    Refine a star position using iterative Gaussian-weighted moments.

    First computes a plain intensity-weighted centroid, then runs three
    iterations re-centring a Gaussian weight function on the current
    estimate.  This converges to sub-pixel accuracy (~0.02-0.05 px)
    much faster than a full nonlinear Gaussian fit, while being nearly
    as accurate for well-sampled stars.

    Parameters
    ----------
    img : 2-D float64 array (bandpass-filtered)
    y0, x0 : int — initial peak position
    half : int — half-size of the fitting patch

    Returns
    -------
    (cx, cy) : float64 — refined sub-pixel position, or None if failed
    """
    H, W = img.shape
    y_lo = max(0, y0 - half)
    y_hi = min(H, y0 + half + 1)
    x_lo = max(0, x0 - half)
    x_hi = min(W, x0 + half + 1)

    patch = img[y_lo:y_hi, x_lo:x_hi].copy()
    ph, pw = patch.shape
    if ph < 5 or pw < 5:
        return None

    yy, xx = np.mgrid[0:ph, 0:pw].astype(np.float64)

    # Clamp negatives (bandpass can have them)
    p = np.maximum(patch, 0)
    total = p.sum()
    if total <= 0:
        return None

    # Iteration 1: plain intensity-weighted centroid
    cy_local = np.sum(yy * p) / total
    cx_local = np.sum(xx * p) / total

    # Gaussian weighting kernel centred on the centroid estimate
    sigma_w = max(half / 2.0, 1.5)

    for _ in range(3):
        r2 = (xx - cx_local)**2 + (yy - cy_local)**2
        weights = np.exp(-r2 / (2 * sigma_w**2))
        wp = p * weights
        wt = wp.sum()
        if wt <= 0:
            break
        cx_local = np.sum(xx * wp) / wt
        cy_local = np.sum(yy * wp) / wt

    cx = float(cx_local + x_lo)
    cy = float(cy_local + y_lo)

    # Sanity: should be close to the initial peak
    if abs(cy - y0) > half * 0.8 or abs(cx - x0) > half * 0.8:
        return None

    return (cx, cy)


def _detect_stars(lum, sigma_thresh=STAR_DETECTION_SIGMA,
                  max_stars=MAX_STARS, border=20, star_fwhm=4.0,
                  min_sharpness=0.35, use_gpu=False):
    """
    Detect point sources in a luminance image.

    Uses a bandpass filter to isolate stellar-scale features from extended
    nebulosity, then finds local maxima and applies a compactness test to
    reject nebula knots.

    Star positions are refined using Gaussian PSF fitting for sub-pixel
    accuracy (~0.02-0.05 px), falling back to intensity-weighted centroid
    if the fit fails.

    The GPU is used only for the expensive gaussian filter step.  All
    peak detection, sharpness testing, and centroid refinement runs on
    CPU in numpy to guarantee identical results regardless of backend.

    Returns (N, 2) float64 array of (x, y) star positions, sorted by
    brightness (brightest first).
    """
    H, W = lum.shape
    img = lum.astype(np.float64)

    # Bandpass: isolate point sources, remove nebulosity
    # GPU accelerates the gaussian filters; result is always numpy
    bandpass = _star_bandpass(img, star_fwhm=star_fwhm, use_gpu=use_gpu)

    # ── Everything below is identical CPU numpy code ──────────────────

    # Noise estimate on bandpass (robust to stars)
    mad = np.median(np.abs(bandpass))
    noise = 1.4826 * mad
    if noise < 1e-6:
        noise = np.std(bandpass)
    if noise < 1e-6:
        return np.empty((0, 2), dtype=np.float64)

    threshold = sigma_thresh * noise

    # Local maximum detection (7×7 neighbourhood)
    local_max = maximum_filter(bandpass, size=7)
    peaks = (bandpass == local_max) & (bandpass > threshold)

    # Exclude borders
    peaks[:border, :] = False
    peaks[-border:, :] = False
    peaks[:, :border] = False
    peaks[:, -border:] = False

    ys, xs = np.where(peaks)
    if len(ys) == 0:
        return np.empty((0, 2), dtype=np.float64)

    # Sort by brightness, pre-filter
    brightness = bandpass[ys, xs]
    order = np.argsort(-brightness)
    if len(order) > max_stars * 5:
        order = order[:max_stars * 5]
    ys, xs, brightness = ys[order], xs[order], brightness[order]

    # Sharpness test + Gaussian centroid refinement
    fit_half = max(int(round(star_fwhm * 1.5)), 4)
    refined = []
    for y0, x0, b in zip(ys, xs, brightness):
        sharpness = _measure_sharpness(bandpass, y0, x0, half=5)
        if sharpness < min_sharpness:
            continue

        result = _gaussian_centroid(bandpass, y0, x0, fit_half)
        if result is not None:
            cx, cy = result
            refined.append((cx, cy, b))

    if not refined:
        return np.empty((0, 2), dtype=np.float64)

    refined.sort(key=lambda t: -t[2])
    if len(refined) > max_stars:
        refined = refined[:max_stars]

    return np.array([(x, y) for x, y, _ in refined], dtype=np.float64)


def detect_stars(lum, use_gpu=True, **kwargs):
    """Detect stars, optionally using GPU for the bandpass filter step."""
    return _detect_stars(lum, use_gpu=use_gpu and _CUPY_AVAILABLE, **kwargs)


# ---------------------------------------------------------------------------
#  Frame sharpness measurement (for auto reference selection)
# ---------------------------------------------------------------------------

def _measure_hfr(lum, stars, half=8):
    """
    Measure the median Half Flux Radius (HFR) of detected stars.

    HFR is the radius that encloses half the total flux in a patch
    around each star.  Smaller HFR = tighter stars = better focus/seeing.

    Parameters
    ----------
    lum : (H, W) float32/64
    stars : (N, 2) array of (x, y)
    half : int — half-size of measurement box

    Returns
    -------
    float — median HFR in pixels.  Returns inf if no stars measured.
    """
    H, W = lum.shape
    hfrs = []

    for sx, sy in stars:
        ix, iy = int(round(sx)), int(round(sy))
        y0, y1 = max(0, iy - half), min(H, iy + half + 1)
        x0, x1 = max(0, ix - half), min(W, ix + half + 1)
        patch = lum[y0:y1, x0:x1].astype(np.float64)

        # Subtract local background (median of patch border)
        bg = np.median(np.concatenate([
            patch[0, :], patch[-1, :], patch[:, 0], patch[:, -1]
        ]))
        patch = np.maximum(patch - bg, 0)
        total = patch.sum()
        if total <= 0:
            continue

        # Compute flux-weighted radial distance from centroid
        cy_local = iy - y0
        cx_local = ix - x0
        yy, xx = np.mgrid[0:patch.shape[0], 0:patch.shape[1]]
        r = np.sqrt((xx - cx_local)**2 + (yy - cy_local)**2)

        # Sort pixels by radius, find radius enclosing 50% flux
        flat_r = r.ravel()
        flat_f = patch.ravel()
        order = np.argsort(flat_r)
        cumflux = np.cumsum(flat_f[order])
        half_flux = total * 0.5

        idx = np.searchsorted(cumflux, half_flux)
        if idx < len(flat_r):
            hfrs.append(float(flat_r[order[idx]]))

    if not hfrs:
        return float('inf')
    return float(np.median(hfrs))


def select_best_reference(frames, stars_per_frame=None,
                          star_fwhm=4.0, detect_sigma=STAR_DETECTION_SIGMA,
                          use_gpu=False, debug_callback=None):
    """
    Select the sharpest frame as the registration reference.

    Measures the median HFR (Half Flux Radius) of stars in each frame
    and returns the index of the frame with the smallest (tightest) stars.

    Parameters
    ----------
    frames : list of (H,W) or (H,W,C) float32 ndarray
    stars_per_frame : list of (N_i, 2) ndarray or None
        Pre-detected stars.  If None, stars are detected on each frame.
    star_fwhm : float
    detect_sigma : float
    use_gpu : bool
    debug_callback : callable or None

    Returns
    -------
    int — index of the best reference frame
    """
    def log(msg):
        if debug_callback:
            debug_callback(msg)

    N = len(frames)
    if N <= 1:
        return 0

    star_kwargs = dict(star_fwhm=star_fwhm, sigma_thresh=detect_sigma)
    hfrs = []

    for i, frame in enumerate(frames):
        lum = _to_luminance(frame)

        if stars_per_frame is not None:
            stars = stars_per_frame[i]
        else:
            stars = detect_stars(lum, use_gpu=use_gpu, **star_kwargs)

        hfr = _measure_hfr(lum, stars)
        hfrs.append(hfr)

    best = int(np.argmin(hfrs))

    log(f"[registration] Frame sharpness (median HFR): "
        f"best={best} ({hfrs[best]:.2f}px), "
        f"worst={int(np.argmax(hfrs))} ({max(hfrs):.2f}px), "
        f"range={min(hfrs):.2f}–{max(hfrs):.2f}px")

    return best


# ---------------------------------------------------------------------------
#  Triangle descriptor matching
# ---------------------------------------------------------------------------

def _build_triangles(stars, max_neighbours=8):
    """
    Build triangle descriptors from star positions.

    For each star, form triangles with its nearest neighbours. Each
    triangle is described by two ratios (d2/d1, d3/d1) where d1 >= d2 >= d3
    are the sorted side lengths. This descriptor is invariant to
    translation, rotation, and (nearly) scale.

    Parameters
    ----------
    stars : (N, 2) array of (x, y) positions
    max_neighbours : int
        Consider only the K nearest neighbours per star to limit combinatorics.

    Returns
    -------
    descriptors : (M, 2) float64 — (ratio1, ratio2) for each triangle
    indices : (M, 3) int — star indices forming each triangle, ordered
        canonically: vertex opposite the longest side first.  This makes
        matched triangles vertex-correspondent by column, so the matcher
        needs no per-match vertex sorting.
    """
    N = len(stars)
    if N < 3:
        return np.empty((0, 2)), np.empty((0, 3), dtype=int)

    # Pairwise distances
    diff = stars[:, np.newaxis, :] - stars[np.newaxis, :, :]  # (N, N, 2)
    dists = np.sqrt((diff ** 2).sum(axis=2))  # (N, N)

    # For each star, find K nearest neighbours
    K = min(max_neighbours, N - 1)

    descriptors = []
    indices = []
    seen = set()

    for i in range(N):
        neighbours = np.argsort(dists[i])[1:K+1]  # skip self
        for j, k in combinations(neighbours, 2):
            tri = tuple(sorted((i, j, k)))
            if tri in seen:
                continue
            seen.add(tri)

            # Side lengths
            a, b, c = tri
            sides = sorted([dists[a, b], dists[a, c], dists[b, c]],
                           reverse=True)
            d1, d2, d3 = sides

            if d1 < 1e-6:
                continue  # degenerate

            descriptors.append((d2 / d1, d3 / d1))
            indices.append(tri)

    if not descriptors:
        return np.empty((0, 2)), np.empty((0, 3), dtype=int)

    descriptors = np.array(descriptors)
    indices = np.array(indices, dtype=int)

    # Canonical vertex order (vectorised): for each triangle, sort the
    # vertices by their opposite side length, descending.  Column k of
    # the result then corresponds between any two matched triangles.
    pa = stars[indices[:, 0]]
    pb = stars[indices[:, 1]]
    pc = stars[indices[:, 2]]
    opp = np.stack([
        np.linalg.norm(pb - pc, axis=1),   # opposite vertex a
        np.linalg.norm(pa - pc, axis=1),   # opposite vertex b
        np.linalg.norm(pa - pb, axis=1),   # opposite vertex c
    ], axis=1)
    order = np.argsort(-opp, axis=1)
    indices = np.take_along_axis(indices, order, axis=1)

    return descriptors, indices


def _match_triangles(desc_ref, idx_ref, stars_ref,
                     desc_src, idx_src, stars_src,
                     tolerance=0.01):
    """
    Match triangle descriptors between reference and source.

    Triangle indices are already in canonical vertex order (built by
    _build_triangles), so vertex correspondence is column-by-column and
    the whole vote accumulation vectorises into one np.add.at.

    Returns (n_ref_stars, n_src_stars) int32 vote-count matrix.
    """
    n_ref = len(stars_ref)
    n_src = len(stars_src)
    votes = np.zeros((n_ref, n_src), dtype=np.int32)

    if len(desc_ref) == 0 or len(desc_src) == 0:
        return votes

    from scipy.spatial import cKDTree

    tree = cKDTree(desc_ref)
    matches = tree.query_ball_point(desc_src, r=tolerance)

    counts = np.fromiter((len(m) for m in matches), dtype=np.intp,
                         count=len(matches))
    if counts.sum() == 0:
        return votes

    ref_tri = np.concatenate([np.asarray(m, dtype=np.intp)
                              for m in matches if m])
    src_tri = np.repeat(np.arange(len(matches), dtype=np.intp), counts)

    # (K, 3) vertex indices for every matched triangle pair; columns
    # correspond by canonical order.
    rv = idx_ref[ref_tri]
    sv = idx_src[src_tri]
    np.add.at(votes, (rv.ravel(), sv.ravel()), 1)

    return votes


def _extract_matches(vote_matrix, min_votes=2):
    """
    From the vote matrix, extract the best one-to-one star matches.

    Uses a greedy approach: take the highest-voted pair, remove both
    stars from the pool, repeat.
    """
    ri_all, si_all = np.nonzero(vote_matrix >= min_votes)
    if len(ri_all) == 0:
        return np.empty((0, 2), dtype=int)

    counts = vote_matrix[ri_all, si_all]
    order = np.argsort(-counts, kind='stable')

    used_ref = set()
    used_src = set()
    matched = []

    for k in order:
        ri, si = int(ri_all[k]), int(si_all[k])
        if ri in used_ref or si in used_src:
            continue
        matched.append((ri, si))
        used_ref.add(ri)
        used_src.add(si)

    return np.array(matched, dtype=int) if matched else np.empty((0, 2), dtype=int)


# ---------------------------------------------------------------------------
#  Transform estimation with RANSAC
# ---------------------------------------------------------------------------

def _fit_similarity(ref_pts, src_pts):
    """
    Fit a similarity transform (rotation + translation + uniform scale)
    from ref_pts -> src_pts using least squares.

    ref_pts, src_pts: (N, 2) arrays of (x, y).

    Returns (2, 3) affine matrix mapping output (ref) coords to source coords:
        src = M[:, :2] @ ref + M[:, 2]

    Or None if the system is underdetermined.
    """
    N = len(ref_pts)
    if N < 2:
        return None

    # Similarity transform: src_x = a*ref_x - b*ref_y + tx
    #                        src_y = b*ref_x + a*ref_y + ty
    # where a = s*cos(θ), b = s*sin(θ)
    # Solve [ref_x, -ref_y, 1, 0] [a]   [src_x]
    #       [ref_y,  ref_x, 0, 1] [b] = [src_y]
    #                              [tx]
    #                              [ty]

    A = np.zeros((2 * N, 4), dtype=np.float64)
    b = np.zeros(2 * N, dtype=np.float64)

    for i in range(N):
        rx, ry = ref_pts[i]
        sx, sy = src_pts[i]
        A[2*i]     = [rx, -ry, 1, 0]
        A[2*i + 1] = [ry,  rx, 0, 1]
        b[2*i]     = sx
        b[2*i + 1] = sy

    try:
        result, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None

    a, bv, tx, ty = result
    M = np.array([[ a, -bv, tx],
                  [bv,   a, ty]], dtype=np.float64)
    return M


def _ransac_similarity(ref_pts, src_pts, threshold=RANSAC_REPROJ_THRESHOLD,
                       max_iterations=500, min_inliers=3):
    """
    RANSAC-based robust similarity transform estimation.

    Parameters
    ----------
    ref_pts, src_pts : (N, 2) arrays, matched star positions
    threshold : float
        Maximum reprojection error (pixels) for inlier classification.
    max_iterations : int
    min_inliers : int

    Returns
    -------
    M : (2, 3) float64 affine matrix, or None if RANSAC fails.
    inlier_mask : (N,) bool array
    """
    N = len(ref_pts)
    if N < 3:
        # min_inliers=3 means fewer points can never yield a valid model.
        return None, np.zeros(N, dtype=bool)

    best_M = None
    best_inliers = np.zeros(N, dtype=bool)
    best_count = 0

    rng = np.random.default_rng(42)

    # Sample size 3 (more robust — a single mismatched pair is far less
    # likely to define a consistent 3-point model than a 2-point one).
    sample_size = 3

    # Adaptive iteration count (standard RANSAC termination): start from
    # the worst case and shrink the required trial count as soon as a model
    # with a high inlier ratio appears.
    #     n_needed = log(1 - p) / log(1 - w**s)
    # where p is the desired probability of sampling an all-inlier set
    # (0.99), w is the current best inlier ratio, and s is the sample size.
    # max_iterations remains a hard ceiling.
    p_success = 0.99
    n_needed = max_iterations

    trial = 0
    while trial < min(max_iterations, int(np.ceil(n_needed))):
        trial += 1
        # Sample the minimal set for the chosen sample size.
        idx = rng.choice(N, size=sample_size, replace=False)
        M = _fit_similarity(ref_pts[idx], src_pts[idx])
        if M is None:
            continue

        # Check scale factor — should be close to 1.0 for astro images
        scale = np.sqrt(M[0, 0]**2 + M[1, 0]**2)
        if scale < 0.95 or scale > 1.05:
            continue

        # Compute reprojection error for all points
        # src_pred = M[:, :2] @ ref.T + M[:, 2:]
        src_pred = (M[:, :2] @ ref_pts.T).T + M[:, 2]
        errors = np.sqrt(((src_pred - src_pts) ** 2).sum(axis=1))

        inliers = errors < threshold
        count = inliers.sum()

        if count > best_count:
            best_count = count
            best_inliers = inliers
            best_M = M

            # Update the adaptive trial budget from the new inlier ratio.
            w = best_count / N
            if w >= 1.0:
                n_needed = 0  # all inliers — no need for more trials
            elif w > 0.0:
                denom = np.log(max(1e-12, 1.0 - w ** sample_size))
                n_needed = np.log(1.0 - p_success) / denom

            # Early exit on a very good fit
            if count >= N * 0.8 and count >= min_inliers:
                break

    if best_count < min_inliers:
        return None, np.zeros(N, dtype=bool)

    # Refit using all inliers
    M_refined = _fit_similarity(ref_pts[best_inliers], src_pts[best_inliers])
    if M_refined is not None:
        best_M = M_refined

    return best_M, best_inliers


# ---------------------------------------------------------------------------
#  Phase correlation fallback
# ---------------------------------------------------------------------------

def _phase_correlate(a, b, xp):
    """Normalised cross-power spectrum. Returns real correlation plane."""
    H, W  = a.shape
    FA    = xp.fft.rfft2(a)
    FB    = xp.fft.rfft2(b)
    cross = FA * xp.conj(FB)
    denom = xp.abs(cross) + 1e-10
    corr  = xp.fft.irfft2(cross / denom, s=(H, W))
    return corr


def _phase_corr_register(ref_lum, src_lum, use_gpu=True,
                         min_peak=MIN_PEAK_HEIGHT,
                         debug_callback=None):
    """
    Phase correlation translation-only registration (fallback).

    Returns (2, 3) affine matrix or None.
    """
    def log(msg):
        if debug_callback:
            debug_callback(msg)

    H, W = ref_lum.shape
    xp = cp if (use_gpu and _CUPY_AVAILABLE) else np
    window = _hann2d(H, W)

    # 3×3 median filter first: frames share fixed-pattern noise (hot
    # pixels survive even imperfect calibration), whose zero-lag
    # correlation otherwise dominates the true stellar peak and locks
    # the fallback onto a false (0, 0) shift with huge SNR.
    from scipy.ndimage import median_filter
    ref_lum = median_filter(ref_lum, size=3)
    src_lum = median_filter(src_lum, size=3)

    # Background subtraction + windowing
    ref_w = (ref_lum - np.median(ref_lum)) * window
    src_w = (src_lum - np.median(src_lum)) * window

    if xp is not np:
        corr = _phase_correlate(cp.asarray(ref_w), cp.asarray(src_w), xp)
        corr_np = cp.asnumpy(corr).astype(np.float32)
        del corr
    else:
        corr_np = np.asarray(_phase_correlate(ref_w, src_w, xp),
                             dtype=np.float32)

    H, W = corr_np.shape
    shifted = np.fft.fftshift(corr_np)

    flat_idx = int(np.argmax(shifted))
    r0, c0 = divmod(flat_idx, W)
    peak_val = float(shifted[r0, c0])
    mean_val = float(np.abs(shifted).mean())
    peak_snr = peak_val / (mean_val + 1e-10)

    # Sub-pixel parabolic refinement
    def parabolic(f, i, n):
        if i <= 0 or i >= n - 1:
            return float(i)
        p, q, r = float(f[(i-1) % n]), float(f[i]), float(f[(i+1) % n])
        d = p - 2*q + r
        return i if abs(d) < 1e-10 else i - 0.5*(r - p)/d

    row_r = parabolic(shifted[:, c0], r0, H)
    col_r = parabolic(shifted[r0, :], c0, W)

    ty = row_r - H // 2
    tx = col_r - W // 2

    log(f"    [phase-corr fallback] peak_snr={peak_snr:.1f}  "
        f"tx={tx:.2f}  ty={ty:.2f}")

    if peak_snr < min_peak:
        log(f"    [phase-corr fallback] SNR too low → identity")
        return None

    M = np.array([[1.0, 0.0, -tx],
                  [0.0, 1.0, -ty]], dtype=np.float64)
    return M


# ---------------------------------------------------------------------------
#  Single-pair registration (star-based with phase-corr fallback)
# ---------------------------------------------------------------------------

def register_pair(ref_lum, src_lum,
                  ref_stars=None,
                  max_rotation_deg=MAX_ROTATION_DEG,
                  min_peak=MIN_PEAK_HEIGHT,
                  star_kwargs=None,
                  use_gpu=True,
                  _src_stars=None,
                  _ref_triangles=None,
                  debug_callback=None):
    """
    Register src_lum to ref_lum.

    Returns 2x3 float64 affine matrix, or None if registration fails.

    Matrix convention (same as gpu_stacking kernel):
        src_coords = M[:, :2] @ dst_coords + M[:, 2]
    """
    def log(msg):
        if debug_callback:
            debug_callback(msg)

    if star_kwargs is None:
        star_kwargs = {}

    H, W = ref_lum.shape

    # ── 1. Star detection ──────────────────────────────────────────────────
    if ref_stars is None:
        ref_stars = detect_stars(ref_lum, use_gpu=use_gpu, **star_kwargs)

    if _src_stars is not None:
        src_stars = _src_stars
    else:
        src_stars = detect_stars(src_lum, use_gpu=use_gpu, **star_kwargs)

    log(f"    stars: ref={len(ref_stars)}, src={len(src_stars)}")

    if len(ref_stars) < 3 or len(src_stars) < 3:
        log(f"    too few stars — falling back to phase correlation")
        return _phase_corr_register(ref_lum, src_lum, use_gpu=use_gpu,
                                    min_peak=min_peak,
                                    debug_callback=debug_callback)

    # ── 2. Triangle matching ───────────────────────────────────────────────
    if _ref_triangles is not None:
        desc_ref, idx_ref = _ref_triangles
    else:
        desc_ref, idx_ref = _build_triangles(ref_stars)
    desc_src, idx_src = _build_triangles(src_stars)

    log(f"    triangles: ref={len(desc_ref)}, src={len(desc_src)}")

    if len(desc_ref) == 0 or len(desc_src) == 0:
        log(f"    no triangles — falling back to phase correlation")
        return _phase_corr_register(ref_lum, src_lum, use_gpu=use_gpu,
                                    min_peak=min_peak,
                                    debug_callback=debug_callback)

    vote_matrix = _match_triangles(desc_ref, idx_ref, ref_stars,
                                    desc_src, idx_src, src_stars,
                                    tolerance=0.01)
    matched = _extract_matches(vote_matrix, min_votes=2)

    log(f"    matched star pairs: {len(matched)}")

    if len(matched) < 3:
        # Retry with looser tolerance
        vote_matrix = _match_triangles(desc_ref, idx_ref, ref_stars,
                                       desc_src, idx_src, src_stars,
                                       tolerance=0.02)
        matched = _extract_matches(vote_matrix, min_votes=2)
        log(f"    matched star pairs (loose): {len(matched)}")

    if len(matched) < 3:
        # RANSAC below needs min_inliers=3, so 2 matches can never succeed.
        log(f"    too few matches — falling back to phase correlation")
        return _phase_corr_register(ref_lum, src_lum, use_gpu=use_gpu,
                                    min_peak=min_peak,
                                    debug_callback=debug_callback)

    ref_matched = ref_stars[matched[:, 0]]
    src_matched = src_stars[matched[:, 1]]

    # ── 3. RANSAC similarity transform ─────────────────────────────────────
    M, inlier_mask = _ransac_similarity(ref_matched, src_matched,
                                        threshold=RANSAC_REPROJ_THRESHOLD)

    if M is None:
        # Soft frames (bad seeing, distorted PSFs): greedy matching pairs
        # nearly every star, but only a handful of pairs are accurate, so
        # RANSAC almost never samples an all-inlier set from the full
        # match list.  Retry with only high-confidence (high-vote) pairs,
        # which restores a workable inlier ratio.
        matched = _extract_matches(vote_matrix, min_votes=10)
        if len(matched) >= 3:
            ref_matched = ref_stars[matched[:, 0]]
            src_matched = src_stars[matched[:, 1]]
            M, inlier_mask = _ransac_similarity(
                ref_matched, src_matched,
                threshold=RANSAC_REPROJ_THRESHOLD)
            log(f"    RANSAC retry with high-vote matches: "
                f"{len(matched)} pairs → "
                f"{'ok' if M is not None else 'failed'}")

    if M is None:
        log(f"    RANSAC failed — falling back to phase correlation")
        return _phase_corr_register(ref_lum, src_lum, use_gpu=use_gpu,
                                    min_peak=min_peak,
                                    debug_callback=debug_callback)

    n_inliers = int(inlier_mask.sum())
    scale = np.sqrt(M[0, 0]**2 + M[1, 0]**2)
    angle = np.degrees(np.arctan2(M[1, 0], M[0, 0]))
    tx, ty = M[0, 2], M[1, 2]

    log(f"    RANSAC: {n_inliers}/{len(matched)} inliers, "
        f"tx={tx:.2f} ty={ty:.2f} rot={angle:.3f}° scale={scale:.5f}")

    # Sanity checks
    if abs(angle) > max_rotation_deg:
        log(f"    rotation {angle:.1f}° exceeds max {max_rotation_deg}° → "
            f"falling back to phase correlation")
        return _phase_corr_register(ref_lum, src_lum, use_gpu=use_gpu,
                                    min_peak=min_peak,
                                    debug_callback=debug_callback)

    if abs(scale - 1.0) > 0.02:
        log(f"    scale {scale:.4f} too far from 1.0 → "
            f"falling back to phase correlation")
        return _phase_corr_register(ref_lum, src_lum, use_gpu=use_gpu,
                                    min_peak=min_peak,
                                    debug_callback=debug_callback)

    return M


# ---------------------------------------------------------------------------
#  Public API
# ---------------------------------------------------------------------------

def identity_matrix():
    return np.array([[1.0, 0.0, 0.0],
                     [0.0, 1.0, 0.0]], dtype=np.float64)


def register_frames(frames,
                    reference_idx=0,
                    max_rotation_deg=MAX_ROTATION_DEG,
                    min_peak=MIN_PEAK_HEIGHT,
                    star_fwhm=4.0,
                    detect_sigma=STAR_DETECTION_SIGMA,
                    force_cpu=False,
                    pre_detected_stars=None,
                    debug_callback=None):
    """
    Register all frames to a reference using star-based matching.

    Parameters
    ----------
    frames : list of (H,W) or (H,W,C) float32 ndarray
    reference_idx : int
    max_rotation_deg : float
        Maximum allowed rotation (frames exceeding this fall back to
        phase correlation or are rejected).
        Set to 0 to restrict to translation only.
    min_peak : float
        Phase-correlation SNR threshold (fallback only).
    star_fwhm : float
        Expected stellar FWHM in pixels.  Controls the bandpass filter
        that separates stars from nebulosity.
    detect_sigma : float
        Star detection threshold in noise sigma units.
    force_cpu : bool
    pre_detected_stars : list of (N_i, 2) ndarray or None
        If provided, skip star detection and use these directly.
        One entry per frame, same order as `frames`.
    debug_callback : callable or None

    Returns
    -------
    list of (2,3) float64 ndarray or None — one per frame.
        Entries are None for frames that could not be registered
        (both star matching and phase correlation failed). The caller
        is responsible for dropping these frames from the stack.
    """
    def log(msg):
        if debug_callback:
            debug_callback(msg)
        else:
            print(msg)

    use_gpu = _CUPY_AVAILABLE and not force_cpu
    log(f"[registration] Star-based registration "
        f"({'GPU' if use_gpu else 'CPU'}), "
        f"{len(frames)} frames, ref={reference_idx}, "
        f"max_rot={max_rotation_deg}°, fwhm={star_fwhm}, σ={detect_sigma}")

    star_kwargs = dict(star_fwhm=star_fwhm, sigma_thresh=detect_sigma)

    ref_lum = _to_luminance(frames[reference_idx])

    if pre_detected_stars is not None:
        ref_stars = pre_detected_stars[reference_idx]
        log(f"[registration] Reference frame {reference_idx}: "
            f"{len(ref_stars)} stars (pre-detected)")
    else:
        ref_stars = detect_stars(ref_lum, use_gpu=use_gpu, **star_kwargs)
        log(f"[registration] Reference frame {reference_idx}: "
            f"{len(ref_stars)} stars detected")

    matrices = [identity_matrix() for _ in frames]
    ref_triangles = _build_triangles(ref_stars)

    for i, frame in enumerate(frames):
        if i == reference_idx:
            log(f"[registration] Frame {i}: reference — identity.")
            continue

        src_lum = _to_luminance(frame)
        log(f"[registration] Frame {i}: registering …")

        src_stars = pre_detected_stars[i] if pre_detected_stars is not None else None

        M = register_pair(ref_lum, src_lum,
                          ref_stars=ref_stars,
                          max_rotation_deg=max_rotation_deg,
                          min_peak=min_peak,
                          star_kwargs=star_kwargs,
                          use_gpu=use_gpu,
                          _src_stars=src_stars,
                          _ref_triangles=ref_triangles,
                          debug_callback=debug_callback)

        if M is None:
            log(f"[registration] Frame {i}: failed — will be dropped.")
            matrices[i] = None
            continue

        angle = np.degrees(np.arctan2(M[1, 0], M[0, 0]))
        log(f"[registration] Frame {i}: "
            f"tx={M[0,2]:.2f} ty={M[1,2]:.2f} rot={angle:.3f}°")
        matrices[i] = M

    if use_gpu and _CUPY_AVAILABLE:
        try:
            cp.get_default_memory_pool().free_all_blocks()
        except Exception:
            pass

    return matrices


def matrices_to_shifts(matrices):
    """Extract (tx, ty) from each affine matrix (for diagnostics).

    Failed frames appear as None in `register_frames` output; they map to
    (nan, nan) so the shift list stays index-aligned with the frames.
    """
    return [(float(M[0, 2]), float(M[1, 2])) if M is not None
            else (float('nan'), float('nan'))
            for M in matrices]


# ---------------------------------------------------------------------------
#  RGB channel alignment (post-stack)
# ---------------------------------------------------------------------------

def align_rgb_channels(image, force_cpu=False, debug_callback=None):
    """
    Align R and B channels to G using phase correlation.

    Corrects sub-pixel chromatic shifts caused by atmospheric dispersion
    or optical chromatic aberration.  Uses G as the reference channel
    (typically sharpest) and computes translation-only corrections for
    R and B via phase correlation.

    Parameters
    ----------
    image : (H, W, 3) float32 ndarray
        Stacked RGB image.
    force_cpu : bool
    debug_callback : callable or None

    Returns
    -------
    (H, W, 3) float32 ndarray — channel-aligned image.
    """
    def log(msg):
        if debug_callback:
            debug_callback(msg)

    if image.ndim != 3 or image.shape[2] != 3:
        log("[rgb_align] Not an RGB image — skipping.")
        return image

    H, W = image.shape[:2]
    xp = cp if (_CUPY_AVAILABLE and not force_cpu) else np

    log(f"[rgb_align] Aligning R,B channels to G "
        f"({'GPU' if xp is not np else 'CPU'})…")

    ref = image[:, :, 1].astype(np.float32)   # G channel = reference
    window = _hann2d(H, W)

    # Background-subtracted + windowed reference
    ref_w = (ref - np.median(ref)) * window

    result = image.copy()

    for ch_idx, ch_name in [(0, 'R'), (2, 'B')]:
        src = image[:, :, ch_idx].astype(np.float32)
        src_w = (src - np.median(src)) * window

        # Phase correlate
        if xp is not np:
            corr = _phase_correlate(cp.asarray(ref_w),
                                    cp.asarray(src_w), xp)
            corr_np = cp.asnumpy(corr).astype(np.float32)
            del corr
        else:
            corr_np = np.asarray(_phase_correlate(ref_w, src_w, xp),
                                 dtype=np.float32)

        # Find peak with sub-pixel refinement
        shifted_corr = np.fft.fftshift(corr_np)
        flat_idx = int(np.argmax(shifted_corr))
        r0, c0 = divmod(flat_idx, W)

        def parabolic(f, i, n):
            if i <= 0 or i >= n - 1:
                return float(i)
            p, q, r = float(f[(i-1) % n]), float(f[i]), float(f[(i+1) % n])
            d = p - 2*q + r
            return i if abs(d) < 1e-10 else i - 0.5*(r - p)/d

        row_r = parabolic(shifted_corr[:, c0], r0, H)
        col_r = parabolic(shifted_corr[r0, :], c0, W)

        ty = row_r - H // 2
        tx = col_r - W // 2

        log(f"[rgb_align] {ch_name} channel: tx={tx:.3f}  ty={ty:.3f}")

        # Apply correction: phase-corr finds the displacement of src
        # relative to ref.  scipy.ndimage.shift(input, (sy, sx)) moves
        # content by (sy, sx), so (ty, tx) is passed directly to undo the
        # detected displacement.
        if abs(tx) > 0.001 or abs(ty) > 0.001:
            from scipy.ndimage import shift as ndimage_shift
            result[:, :, ch_idx] = ndimage_shift(
                src, (ty, tx), order=1, mode='constant', cval=0.0)

    if xp is not np:
        try:
            cp.get_default_memory_pool().free_all_blocks()
        except Exception:
            pass

    log("[rgb_align] Channel alignment complete.")
    return result
