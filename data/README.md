# Data

**Dataset:** Bank Marketing (bank-full version), UCI Machine Learning Repository
**Licence:** CC BY 4.0
**Citation:** Moro, S., Rita, P., & Cortez, P. (2014). *Bank Marketing* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5K306

The raw data is not committed to the repository. To download it, run from the repository root:

```
python src/download_data.py
```

This saves `data/raw/bank-full.csv` (45,211 rows, 17 columns, semicolon-separated).

The data covers direct-marketing phone calls made by a Portuguese bank between May 2008 and November 2010. The target `y` records whether the client subscribed to a term deposit.
