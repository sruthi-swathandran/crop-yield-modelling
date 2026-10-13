# Predicting county corn yields from satellite time series

Predicting US county-level corn yields from MODIS vegetation indices, 2008 to
2021, with an honest evaluation of how much accuracy a random train-test split
invents.

Status: **in progress**

---

## The question this is really about

Most satellite yield models report a single accuracy figure from a random
train-test split. On spatial data that figure is inflated, and often badly.
Neighbouring counties share weather, soil and management practice, so a random
split puts near-identical observations on both sides of the fold and the model
is partly remembering rather than generalising.

An operational user never asks the question a random split answers. They ask
about a season that hasn't happened yet, or a region the model has never seen,
or both.

So this project trains one model and evaluates it four ways:

| Scheme | The question it answers |
|---|---|
| Random k-fold | What most papers report |
| Leave one year out | Can it handle an unseen season |
| Leave one region out | Can it handle an unseen place |
| Leave out both | What an operational user needs |

The gap between the first and the last is the finding.

---

## Data

| Source | What | Period |
|---|---|---|
| USDA NASS Quick Stats | County corn yields, bushels per acre | 2008 to 2021 |
| USDA Cropland Data Layer | Annual corn mask, 30 m | 2008 to 2021 |
| MODIS MOD13Q1 (Terra) | NDVI, 250 m, 16-day composite | 2008 to 2021 |
| NASA POWER | Daily shortwave radiation, temperature, rainfall | 2008 to 2021 |
| US Census cartographic boundaries | County boundaries | 2023 |

Nothing in `data/` is committed. Every source is public and the fetch scripts in
`src/` reproduce it.

### Why it stops in 2021

The Cropland Data Layer is read from Microsoft's Planetary Computer, whose copy
ends in 2021. USDA itself has published through 2025, so the limit is the
hosting rather than the data, and later years can be downloaded directly.

There is a second reason to treat later years separately. Terra, the satellite
carrying the MODIS sensor used here, had its last orbit correction in February
2020 and has drifted to an earlier overpass time since, leaving its constellation
in 2022. Earlier overpasses mean different illumination, so NDVI after 2022 is
not strictly comparable with the years the model learns from. 2008 to 2021 is
fourteen seasons on a stable orbit, including the 2012 drought.

### Why MODIS and not Sentinel-2

The Cropland Data Layer stops at 2021. Sentinel-2 starts in 2017. Together those
leave five seasons.

Five is not enough. The model has to learn how yield responds to weather, and
weather varies between years rather than within them. Worse, 2012 falls outside
that window, and 2012 is the most informative year in the dataset: mean yield
across these eight states fell from 143 bushels per acre to 111. A model that has
never seen a drought cannot be tested on one.

MODIS reaches back to 2008 and gives fourteen seasons including that drought.

The resolution Sentinel-2 would buy is resolution this question throws away.
Both get averaged to a county containing millions of pixels. Ten metres matters
for a field-level model and not for this one.

There is also a practical weight. Sentinel-2 is 625 times more pixels per unit
area, and MOD13Q1 arrives already composited and cloud-screened, where Sentinel-2
needs its own cloud masking and gap-filling from irregular revisits.

Fourteen seasons including a drought, at a resolution the question can use,
against five seasons without one at a resolution it cannot.

---

## Three models, one evaluation

Every model in this project is evaluated on identical counties, years and
cross-validation folds, so differences between them are differences between
the models rather than between the tests.

| Model | Inputs | Fitted parameters |
|---|---|---|
| Gradient boosting | MODIS NDVI curve and phenology | Hundreds |
| Light use efficiency | MODIS fAPAR, NASA POWER radiation | One or two |
| Gradient boosting, Sentinel-2 | Sentinel-2 NDVI, Iowa 2017 to 2021 | Hundreds |

### The semi-physical model

Monteith's light use efficiency framework, the standard middle ground between
fitted and process-based yield models:

```
yield = HI x e x sum(fAPAR x PAR)
```

Absorbed radiation accumulated across the season, converted to biomass by a
radiation use efficiency `e`, then to grain by a harvest index `HI`. fAPAR comes
from NDVI through an empirical relationship, which is what makes it semi- rather
than fully physical.

Radiation use efficiency and harvest index come from published values for
maize, cited, rather than being fitted. At most one scalar is calibrated on
training folds to absorb systematic bias.

**The hypothesis:** the gradient boosting model wins on random folds, and the
light use efficiency model degrades less when held out spatially. A model with
hundreds of fitted parameters learns the relationships of the region it was
trained on; a model whose few parameters mean something physical should carry
further. If that holds it answers a question practitioners actually face, which
is what to reach for when you need to work somewhere you have no training data.

It may not hold. The light use efficiency model sees greenness and light, not
water. Drought during silking can cost a third of a crop before the canopy shows
it, so 2012 is where this model is most likely to fail, and that failure will be
reported rather than tuned away.

---

## Planned extension: does 10 metre imagery predict yield any better?

2017 to 2021 is covered by both sensors. That overlap makes a real ablation
possible, run on the same counties, the same years and the same model:

> Holding everything else constant, does a Sentinel-2 feature set predict
> county corn yield better than a MODIS one?

Nobody has a strong prior on the answer. The case for yes is that 10 metre
pixels resolve field boundaries, so the crop mask is sharper and fewer mixed
pixels dilute the signal. The case for no is that county aggregation averages
away everything the finer grid resolved, and Sentinel-2's irregular cloud-limited
revisits produce a worse time series than MODIS's clean composites.

It reuses this pipeline rather than needing a new one, and it answers a question
that practitioners actually argue about. To be built once the MODIS version is
finished and evaluated, not alongside it.

### Scope and its limits

**Iowa only, 99 counties, 2017 to 2021.** Sentinel-2 is 625 times the pixels of
MODIS per unit area and arrives as individual scenes rather than clean
composites, so it needs cloud masking from the scene classification layer and
compositing onto a regular interval before it is comparable at all. Across the
full study area that is a very large job. Iowa is the cleanest corn signal in
the dataset, which makes it the fairest place to ask whether resolution helps,
and MODIS is evaluated on exactly the same subset so the comparison stays
like for like.

**From 2017 to 2021 the crop mask cannot get sharper than 30 metres.** It comes
from the Cropland Data Layer, which was 30 metre data until 2023. Part of the
case for Sentinel-2 is that finer pixels give a cleaner corn mask, and in these
years that part of the argument is capped by the mask source no matter how fine
the imagery is. Whatever advantage Sentinel-2 shows here comes from the
vegetation signal, not from sharper crop boundaries.

**2024 and 2025 remove that cap.** USDA moved the Cropland Data Layer to 10 metres
from 2024, built partly from Sentinel-2 itself. Repeating the comparison on those
two seasons, with a 10 metre mask for Sentinel-2 and the same mask averaged to
250 metres for MODIS, tests the sharper-mask argument directly. Two seasons is
too few to train on, so these years are used for evaluation only, with models
trained on the earlier window.

---

## Planned extension: seasons the model has never seen

2022 to 2025 are held back as a test set. Every model is trained on 2008 to 2021
and then asked to predict four later seasons, which is the question an
operational user is actually asking.

Three things change for those years and each is handled rather than ignored:

- **The crop mask** is downloaded from USDA's CroplandCROS rather than the
  Planetary Computer. 2024 and 2025 come at 10 metres and are averaged onto the
  MODIS grid exactly as the 30 metre years are.
- **Yields** are fetched again from NASS, since the original fetch stopped at
  2023.
- **Terra's orbit drift** is measured before it is trusted. NDVI over fixed,
  stable cropland is compared before and after 2022, and if the drift shifts
  the values, the shift is reported alongside every 2022 to 2025 result.

NASA plans to wind down Terra and Aqua from late 2026 and points users to
VIIRS, the successor sensor. Carrying this pipeline over to VIIRS is the natural
next step and is out of scope here.

---

## Setup

```
conda env create -f environment.yml
conda activate cropyield
```

A free NASS Quick Stats API key is needed, requested at
<https://quickstats.nass.usda.gov/api>. Put it in `.env` as:

```
NASS_API_KEY=your_key_here
```

Then run `src/` in numerical order.

---

## Known traps, documented before they bite

**Yield trends upward over time** through breeding and management, independent
of weather. A model trained across fourteen years can learn that trend and appear
to predict yield while really predicting the year. Handled explicitly rather
than hoped about.

**Counties are not independent observations.** This is the subject of the
project rather than a caveat on it.

**The crop mask changes annually**, so the pixels averaged for a county in 2011
are not the pixels averaged in 2019. That is correct behaviour and worth stating,
since it means the feature is "NDVI over this year's corn" rather than "NDVI
over a fixed area".
