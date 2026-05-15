import numpy as np
import random
from collections import Counter
from math import comb

class CCCFBI:
    """
    3C-FBI: Combinatorial Convolution-based Circle Fitting in Blurry Images
    """
    def __init__(self, eps=20, n_tri=5000, n_peaks=5, tau=1):
        self.eps = eps
        self.n_tri = n_tri
        self.n_peaks = n_peaks
        self.tau = tau

    def _solve_circle(self, p1, p2, p3):
        """Eq. circle_formula: Closed-form circle parameters from 3 points."""
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3

        # Determinant calculation (Filter 1: Collinearity)
        D = 2 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
        if abs(D) < 1e-8:
            return None

        # Closed form for center (x, y) and radius r
        s1, s2, s3 = x1**2 + y1**2, x2**2 + y2**2, x3**2 + y3**2
        
        xc = (s1 * (y2 - y3) + s2 * (y3 - y1) + s3 * (y1 - y2)) / D
        yc = (s1 * (x3 - x2) + s2 * (x1 - x3) + s3 * (x2 - x1)) / D
        rc = np.sqrt((x1 - xc)**2 + (y1 - yc)**2)
        
        return (xc, yc, rc)

    def fit(self, image, r_range):
        """
        Main algorithm execution
        :param image: Binary edge image (2D numpy array)
        :param r_range: Tuple (r_min, r_max)
        :return: (x_hat, y_hat, r_hat)
        """
        y_max, x_max = image.shape
        r_min, r_max = r_range

        # Step 1: Edgel extraction (E)
        # Using nonzero() to get indices of "black" pixels (assuming edge=255 or 1)
        edgel_y, edgel_x = np.nonzero(image)
        E = list(zip(edgel_x, edgel_y))
        num_edgels = len(E)

        if num_edgels < 3:
            return None

        # Step 2: Triplet sampling and voting
        V = Counter()
        # Sample min(N_tri, |E| choose 3)
        actual_n = min(self.n_tri, comb(num_edgels, 3))
        
        # Use a set for sampling to ensure distinct triplets if N is small
        # but random.sample is sufficient for N_tri = 5000
        for _ in range(actual_n):
            p1, p2, p3 = random.sample(E, 3)
            
            res = self._solve_circle(p1, p2, p3)
            if res is None: continue  # Filter 1: Collinear
            
            x, y, r = res
            
            # Filter 2: Center tolerance
            if not (-self.eps <= x <= x_max + self.eps and 
                    -self.eps <= y <= y_max + self.eps):
                continue
            
            # Filter 3: Radius range
            if not (r_min <= r <= r_max):
                continue
            
            # Vote in sparse map
            cell = (int(round(x)), int(round(y)), int(round(r)))
            V[cell] += 1

        if not V:
            return None

        # Step 3: Top-N peak selection (P)
        # Counter.most_common handles ties by first occurrence order
        P = [item[0] for item in V.most_common(self.n_peaks)]

        # Step 4: Localized cube scoring
        localized_results = []
        for p_i in P:
            w_total = 0
            weighted_centroid = np.array([0.0, 0.0, 0.0])
            
            # (2*tau + 1)^3 cube search
            for dx in range(-self.tau, self.tau + 1):
                for dy in range(-self.tau, self.tau + 1):
                    for dr in range(-self.tau, self.tau + 1):
                        neighbor = (p_i[0] + dx, p_i[1] + dy, p_i[2] + dr)
                        votes = V.get(neighbor, 0)
                        if votes > 0:
                            w_total += votes
                            weighted_centroid += votes * np.array(neighbor)
            
            p_hat_i = weighted_centroid / w_total
            localized_results.append((w_total, p_hat_i))

        # Step 5: Final estimate (argmax W_pi)
        # max() returns the first occurrence in case of ties
        best_peak = max(localized_results, key=lambda x: x[0])
        
        return best_peak[1] # Returns (x_hat, y_hat, r_hat)

# --- Quick Test ---
if __name__ == "__main__":
    # Create synthetic blurred edge data
    test_img = np.zeros((100, 100))
    # Draw a circle at (50, 50) with radius 30
    for a in np.linspace(0, 2*np.pi, 200):
        tx = int(50 + 30*np.cos(a) + np.random.normal(0, 0.5))
        ty = int(50 + 30*np.sin(a) + np.random.normal(0, 0.5))
        if 0 <= tx < 100 and 0 <= ty < 100:
            test_img[ty, tx] = 1

    detector = CCCFBI(n_tri=8000)
    result = detector.fit(test_img, r_range=(20, 40))
    
    if result is not None:
        print(f"Final Estimate: X={result[0]:.3f}, Y={result[1]:.3f}, R={result[2]:.3f}")