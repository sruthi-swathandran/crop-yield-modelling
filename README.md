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

Sentinel-2 is 10 metres and starts in 2017, giving eight seasons. MODIS is 250
metres and starts in 2000, giving more than twenty.

Aggregating to county level discards the resolution advantage entirely: a county
averages millions of pixels either way. Choosing the coarser sensor buys twelve
extra years of training data and costs nothing that this problem can use.

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
