# Predicting county corn yields from satellite time series

Predicting US county-level corn yields from MODIS vegetation indices, 2008 to
2023, with an honest evaluation of how much accuracy a random train-test split
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
| USDA NASS Quick Stats | County corn yields, bushels per acre | 2008 to 2023 |
| USDA Cropland Data Layer | Annual corn mask | 2008 to 2023 |
| MODIS MOD13Q1 | NDVI and EVI, 250 m, 16-day composite | 2008 to 2023 |
| US Census TIGER | County boundaries | |

Nothing in `data/` is committed. Every source is public and the fetch scripts in
`src/` reproduce it.

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
of weather. A model trained across sixteen years can learn that trend and appear
to predict yield while really predicting the year. Handled explicitly rather
than hoped about.

**Counties are not independent observations.** This is the subject of the
project rather than a caveat on it.

**The crop mask changes annually**, so the pixels averaged for a county in 2011
are not the pixels averaged in 2019. That is correct behaviour and worth stating,
since it means the feature is "NDVI over this year's corn" rather than "NDVI
over a fixed area".
