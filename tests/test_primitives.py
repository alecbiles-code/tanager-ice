"""
Synthetic, known-answer tests for the numerical primitives. No network, no data.
Run: python tests/test_primitives.py
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tanager_ice import spectral as sp
from tanager_ice import separability as sep
from tanager_ice import uncertainty as unc

rng = np.random.default_rng(0)
wl = np.arange(380, 2500, 5.0)          # ~5 nm sampling like Tanager
ok = []

def check(name, cond, detail=""):
    ok.append(cond)
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")

# 1. Continuum removal: a pure linear spectrum lies ON its continuum -> CR == 1.
lin = 2.0 + 0.001 * wl
_, cr = sp.continuum_removed(lin, wl, 900, 1100)
check("continuum removal flattens a line", np.allclose(cr, 1.0, atol=1e-9),
      f"max|CR-1|={np.max(np.abs(cr-1)):.2e}")

# 2. Band depth: carve a triangular notch of known depth at 1030 nm.
spec = 1.0 + 0.0 * wl
c = sp.nearest_index(wl, 1030)
notch = np.clip(1 - np.abs(np.arange(len(wl)) - c) / 8.0, 0, 1) * 0.30
spec2 = spec - notch
d = sp.band_depth(spec2, wl, 1030, 960, 1080)
check("band depth recovers 0.30 notch", abs(d - 0.30) < 1e-6, f"depth={d:.4f}")

# 3. Scaled band area increases monotonically with feature width (grain proxy).
def make_feature(width_pts, depth=0.3):
    s = np.ones_like(wl)
    return s - np.clip(1 - np.abs(np.arange(len(wl)) - c) / width_pts, 0, 1) * depth
areas = [sp.scaled_band_area(make_feature(w), wl, 960, 1080) for w in (4, 8, 12)]
check("band area monotonic in width", areas[0] < areas[1] < areas[2],
      f"areas={[round(a,2) for a in areas]}")

# 4. CIBR < 1 when a 940 nm absorption notch is present, ~1 when absent.
flat = np.ones_like(wl)
ci_flat = sp.cibr(flat, wl)
g = np.exp(-0.5 * ((wl - 940) / 8) ** 2) * 0.4
ci_abs = sp.cibr(flat - g, wl)
check("CIBR ~1 with no vapour", abs(ci_flat - 1.0) < 1e-6, f"cibr={ci_flat:.4f}")
check("CIBR <1 with vapour band", ci_abs < 0.8, f"cibr={ci_abs:.4f}")

# 5. DOS-Rayleigh: dark anchor (near-zero surface) sets A; bright target flattens.
#    DOS assumes the dark object's surface radiance ~ 0, so the anchor must be
#    ~path-only. A bright flat target then flattens after subtracting A*(ref/wl)^4.
A_true = 5.0
add = A_true * (440.0 / wl) ** 4
dark = np.tile(np.full_like(wl, 0.01), (50, 1)) + add    # ~0 surface + path
A_est = sp.dos_rayleigh_estimate(dark, wl, ref_nm=440, dark_pct=50)
bright = np.ones_like(wl) + add                          # surface=1 + same path
corr = sp.dos_rayleigh_correct(bright, wl, A_est, ref_nm=440)
check("DOS-Rayleigh recovers amplitude", abs(A_est - A_true) < 0.02, f"A={A_est:.4f}")
check("DOS-Rayleigh flattens bright target", np.allclose(corr, 1.0, atol=0.02),
      f"max dev={np.max(np.abs(corr-1)):.2e}")

# 6. Sediment index rises when we redden a spectrum; DOS removal changes a
#    Rayleigh-contaminated slope toward the true (flat) value.
clean = np.ones_like(wl)
redder = clean.copy(); redder[wl < 600] *= 0.6           # darker blue -> 'dirtier'
si_clean = sp.sediment_index(clean, wl)
si_red = sp.sediment_index(redder, wl)
check("sediment index rises with reddening", si_red > si_clean + 0.1,
      f"clean={si_clean:.3f} red={si_red:.3f}")

# 7. Spectral angle: identical -> 0, orthogonal -> pi/2.
a = rng.random(20); b = a.copy()
check("SAM identical=0", sep.spectral_angle(a, b) < 1e-6)
e1 = np.zeros(20); e1[0] = 1; e2 = np.zeros(20); e2[1] = 1
check("SAM orthogonal=pi/2", abs(sep.spectral_angle(e1, e2) - np.pi/2) < 1e-8)

# 8. Jeffries-Matusita: identical distributions -> ~0, far-apart -> ~2.
Xa = rng.normal(0, 1, (200, 30))
Xb_same = rng.normal(0, 1, (200, 30))
Xb_far = rng.normal(8, 1, (200, 30))
jm_same = sep.jeffries_matusita(Xa, Xb_same, pca_k=8)
jm_far = sep.jeffries_matusita(Xa, Xb_far, pca_k=8)
check("JM identical ~0", jm_same < 0.3, f"jm={jm_same:.3f}")
check("JM far ~2", jm_far > 1.9, f"jm={jm_far:.3f}")

# 9. Conformal regression interval achieves >= 1-alpha coverage out of sample.
alpha = 0.1
cal_true = rng.normal(0, 1, 2000); cal_pred = np.zeros_like(cal_true)
q = unc.regression_interval(cal_pred, cal_true, alpha)
test_true = rng.normal(0, 1, 5000); test_pred = np.zeros_like(test_true)
cov = unc.empirical_coverage((test_pred - q, test_pred + q), test_true, mode="reg")
check("conformal regression coverage >= 90%", cov >= 0.89, f"cov={cov:.3f}")

# 10. Conformal classification sets achieve >= 1-alpha coverage.
K = 4
def softmax(z): e = np.exp(z - z.max(1, keepdims=True)); return e / e.sum(1, keepdims=True)
cal_y = rng.integers(0, K, 2000)
cal_logits = rng.normal(0, 1, (2000, K)); cal_logits[np.arange(2000), cal_y] += 1.5
cal_p = softmax(cal_logits)
test_y = rng.integers(0, K, 5000)
test_logits = rng.normal(0, 1, (5000, K)); test_logits[np.arange(5000), test_y] += 1.5
test_p = softmax(test_logits)
qh, sets = unc.classification_sets(cal_p, cal_y, test_p, alpha)
cov_c = unc.empirical_coverage(sets, test_y, mode="class")
check("conformal classification coverage >= 90%", cov_c >= 0.89,
      f"cov={cov_c:.3f} avg|set|={sets.sum(1).mean():.2f}")

print(f"\n{sum(ok)}/{len(ok)} checks passed")
sys.exit(0 if all(ok) else 1)
