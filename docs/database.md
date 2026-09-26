# Dataset Documentation

## Dataset

**Name:** Telecom Churn Dataset

**Source:** Mendeley Data

**Dataset URL:** https://data.mendeley.com/datasets/nrb55gr66h/1

**License:** CC BY 4.0

**Domain:** Telecommunications

**Customer type:** Business / B2B customers

## Dataset purpose

The dataset contains customer and service information from a telecommunications operator and includes a binary churn indicator.

The project uses this dataset to develop a customer churn prediction and retention-risk system.

## Local dataset

The downloaded source file is stored as:

`data/raw/telecom_churn.csv`

The raw dataset is treated as immutable.

No manual cleaning, transformation, deletion, or modification should be performed on the raw file.

## Initial observed dimensions

Rows: 8,453

Columns: 14

## Target

The target variable is:

`CHURN`

The target represents whether the customer churned.

## Observed features

* `PID`
* `CRM_PID_Value_Segment`
* `EffectiveSegment`
* `Billing_ZIP`
* `KA_name`
* `Active_subscribers`
* `Not_Active_subscribers`
* `Suspended_subscribers`
* `Total_SUBs`
* `AvgMobileRevenue`
* `AvgFIXRevenue`
* `TotalRevenue`
* `ARPU`

## Reproducibility

The exact raw file used for model development should be identified using a cryptographic SHA-256 checksum.

The checksum should be recorded after the raw dataset has been downloaded and before any processing occurs.

If the source dataset is replaced or updated, the new file must receive a new checksum and dataset version.

## Important data-quality observations

The raw CSV contains a trailing whitespace character in the column name:

`AvgMobileRevenue `

This should not be manually corrected in the raw file.

Column normalization will be performed programmatically in the data-ingestion/validation pipeline.

## Dataset checksum

SHA-256:

`36509d631e29a70e4a6e01c0deef42f47d35c112ea29d94b21b67b0a2e26c523`