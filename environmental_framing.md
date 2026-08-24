# Watching the Arctic Coast Melt Where Our Sensors Cannot

### A Tanager hyperspectral demonstration of cryosphere surface-state retrieval at a latitude and in spectral regions that operational Earth-observation systems miss

*Planet Tanager Open Data Competition: environmental framing*

---

## The blind spot

Every spring, the Arctic coast begins to melt. The timing and pattern of that melt (where liquid water first appears on snow and sea ice, how the snowpack coarsens as it ages) governs how much sunlight the surface absorbs, which in turn governs how fast the melt accelerates. This is the snow- and ice-albedo feedback, one of the central amplifiers of Arctic warming. The variables that drive it are physical surface properties: **snow grain size** and the **presence of liquid water**.

Here is the problem: at the latitudes where this matters most, we largely cannot see those properties from space.

The reason is a gap between two kinds of instrument, and it is a gap this submission makes concrete:

- **The hyperspectral sensor that could see surface state, EMIT, physically cannot reach the high Arctic.** EMIT flies on the International Space Station, whose orbit is inclined about 52°. Its imaging swath does not extend to the Arctic coast. The single best spaceborne imaging spectrometer for surface material is structurally absent from the region where melt onset begins.
- **The sensors we *do* fly at those latitudes, multispectral systems like Sentinel-2, cannot see the surface features that reveal melt and grain size.** Sentinel-2 has no spectral band between 958 and 1565 nm. Both diagnostic ice-absorption features, liquid water near 970 nm and grain size near 1030 nm, fall inside that gap. Sentinel-2 sees the Arctic coast clearly, at fine spatial resolution, and is blind to the two properties that matter here.

So there is a real hole in Earth observation: the right *spectral* tool is in the wrong *orbit*, and the right *orbit* carries the wrong *spectral* tool.

**Tanager fills exactly this hole.** It is an imaging spectrometer (the same measurement class as EMIT, 426 contiguous bands at ~5 nm) flown in a polar orbit that reaches the Arctic coast. This submission uses a single Tanager scene to demonstrate what becomes visible when you finally point a spectrometer at the place, and the season, where the current system is blind.

---

## The scene

**Baffin Island coast, Nunavut, 73.7° N, 6 June 2025.** A June acquisition at melt-season onset, over a coastline that packs three cryosphere regimes into one frame:

- **snow-covered mountain terrain** (the DEM confirms 0-278 m relief, 44% of the scene is land),
- **sea ice** offshore, from consolidated floes to a broken marginal-ice field threaded with leads, and
- **open water** in the leads and beyond the ice edge.

One scene, one instant, a full land-ice-ocean cryosphere transect, at 73.7° N, which clears EMIT's reach by roughly 21° of latitude. This is not a scene EMIT could have collected. It is the kind of scene the competition exists to identify.

---

## What Tanager resolves here

We ran three retrievals, each built to be honest about what it can and cannot claim. The through-line: **Tanager resolves surface *material state*, not merely surface *type*.**

### 1. A grain-size-sensitive proxy, with calibrated per-pixel uncertainty

Snow and ice reflectance near 1030 nm deepens as grains coarsen, the basis of the established Nolin-Dozier scaled-band-area method. We retrieve this proxy per pixel across all ice-bearing surfaces, and, critically, we attach a **calibrated uncertainty to every pixel**. Using Planet's shipped per-band reflectance-uncertainty cube, we propagate a physics-based error onto the proxy, then wrap it in a normalized split-conformal interval whose empirical coverage on this scene matches its 90% target. The proxy separates cleanly by surface class: metamorphosed sea ice and mountain snow carry a coarser signature than thin or mixed ice, and an internal operator check confirms the retrieval is self-consistent.

We are deliberate about the claim: this is a **grain-size-*sensitive* proxy**, not an absolute grain radius. It co-varies with grain size but also with brightness and impurity, and an absolute radius would require a radiative-transfer inversion we do not attempt. What we demonstrate is a *spatially resolved, uncertainty-quantified relative field*: the thing you need before any absolute retrieval is worth attempting.

### 2. A surface-liquid-water signal, separated from the mixing artifact that would fake it

Liquid water absorbs near 970 nm, so this feature is the natural melt indicator, but it carries a trap that has defeated naive analyses: the same feature responds to *any* water in a pixel, including sub-pixel open water in leads and melt ponds. A raw "melt map" of a marginal ice zone is often just a water-fraction map.

We built a guard against exactly this. Each pixel is spectrally unmixed into ice and open-water fractions; the liquid-water signal is then interpreted only where the ice fraction is high, and tested for independence from water fraction. On this scene the liquid-water signal is **statistically independent of sub-pixel water fraction** (correlation -0.09) and varies across pure-ice pixels at roughly four times the sensor noise floor. In other words: the signal is a genuine property of the ice surface, not a mixing artifact, and we can prove it, because the guard was validated in both directions on controlled data (it detects planted melt and correctly reports a null when none is present).

We frame this precisely as a **surface-liquid-water signal consistent with early melt** at a June Arctic coast: spatially resolved, and demonstrably not a sub-pixel-water artifact. The unmixing guard is itself a methodological contribution: it is the discipline that makes a melt claim from a marginal ice zone defensible at all.

### 3. A surface classification with honest confidence

We turn the scene into a five-class surface map (snow terrain, sea ice, and mixed/melt/water classes) and attach conformal prediction to it, so that ambiguous pixels can carry a *set* of plausible classes rather than a falsely confident single label. The class map is the scaffold on which the material retrievals are stratified.

---

## Why this needs Tanager specifically: the capability gap, quantified

The competition's core question is which hyperspectral scenes are worth releasing. Our strongest answer is a direct, quantified demonstration of what is lost without hyperspectral sampling.

We simulated Sentinel-2 by convolving each Tanager spectrum to Sentinel-2's published spectral response, holding the scene fixed so we isolate the *spectral* axis. The result splits cleanly by task:

- **Grain size and melt: Sentinel-2 retains 0% of the signal, not "less," but zero, to numerical precision.** The 970 nm and 1030 nm features lie entirely inside Sentinel-2's 958-1565 nm gap. Interpolating across that gap yields a straight line, whose band area is exactly zero by construction. This is not Sentinel-2 performing poorly; it is a **structural capability gap**: the retrieval requires a spectral region the sensor does not measure at all.
- **Broadband classification: near-parity, and we concede Sentinel-2's likely advantage.** For the surface-type map, Tanager and simulated Sentinel-2 reach comparable accuracy, and Sentinel-2's finer native spatial resolution (10-20 m vs Tanager's 33 m) would likely *exceed* Tanager on this task in practice.

That honest split is the point. **Hyperspectral is not uniformly "better." It is *categorically different*: it resolves surface material state (the grain size and liquid water that drive the albedo feedback) that multispectral sensors are structurally blind to, at any spatial resolution.** Where surface *type* is the question, fly the cheaper multispectral sensor. Where surface *state* is the question, as it is for melt onset and snow aging, only the spectrometer will do. That is the argument for releasing scenes like this one.

---

## What we do *not* claim, and why that is a strength

There is no field validation in this work. No in-situ grain-size or liquid-water measurement was coincident with this scene, and none exists for essentially any open Tanager acquisition. Rather than paper over this, we designed the entire study around it:

- Every retrieval reports **calibrated precision, not accuracy.** Our uncertainty intervals are honest statements of self-consistency and sensor noise, explicitly *not* claims of agreement with ground truth.
- Our capability claims are **structural**: that Sentinel-2 cannot sample 1030 nm is a fact about band placement, not a contested measurement.
- We validated every method on controlled synthetic data, in both directions, before trusting it on the real scene.

This is the mature posture for open pre-validation data: establish what the sensor can and cannot *resolve*, with calibrated confidence, so that the field campaigns which would establish *accuracy* can be aimed at the scenes and phenomena that merit them. **Choosing those scenes is precisely what this competition's prize enables**, and a submission that demonstrates calibrated capability, rather than overclaiming accuracy, is the one that can actually guide that choice.

---

## Why release this scene, and scenes like it

This single Baffin acquisition demonstrates:

1. **A latitude EMIT cannot reach** (73.7° N), extending imaging-spectrometer science into the high Arctic that the current workhorse structurally misses.
2. **A season that matters**, melt onset, where surface state is changing fastest and is most consequential for climate feedback.
3. **A land-ice-ocean transect in one frame**, letting a single scene exercise snow, sea ice, and water retrievals together.
4. **Material-state retrievals that multispectral sensors cannot reproduce**, with the capability gap quantified rather than asserted.
5. **Calibrated, per-pixel uncertainty throughout**, built on Planet's own uncertainty product, a template other users can adopt directly.

A high-Arctic, melt-season, coastal-transect hyperspectral scene is a scarce and scientifically rich asset. If the goal is to advance cryosphere research where the observing system is currently blind, this is exactly the kind of scene worth releasing, and this submission is a working prototype of the science it would unlock.

---

*All retrievals, the atmospheric-correction trust analysis, the DEM coregistration and topographic handling, the uncertainty calibration, and the sensor-degradation experiment are implemented in a scene-agnostic, tested Python pipeline accompanying this submission, runnable end-to-end from the public STAC item.*
