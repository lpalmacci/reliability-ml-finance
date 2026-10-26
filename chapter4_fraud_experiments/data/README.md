# Data of Chapter 4

The two fraud datasets used in Chapter 4, as distributed by their sources. The loader (`fraud/loader.py`) checks the
SHA-256 and the counts of each file and stops if they differ; the cleaning (removed rows and columns) is done by the
code, not in these files.

Before the first run, decompress ULB (GitHub does not accept files above 100 MB):

```bash
gunzip -k data/ulb_creditcard.csv.gz        # -> data/ulb_creditcard.csv (144 MB, ignored by git)
```

| File | SHA-256 | Rows / frauds |
|---|---|---|
| `fraud_oracle.csv` | `d7dab6086aa186fc82c607d50919067a1599cea45afb4965511fd7da9e935e09` | 15,420 / 923 (15,419 / 923 after the cleaning in `fraud/loader.py`) |
| `ulb_creditcard.csv.gz` | `68799ece2dc50b1c9873ba214b48fe63914b2c239d1067c3b27c4c423bbfaaa4` | |
| `ulb_creditcard.csv` (decompressed) | `76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89` | 284,807 / 492 |

`ulb_creditcard.csv.gz` was written with `gzip -9 -n`, so compressing the CSV again gives the same file.

## ULB: credit card transactions

- **Source:** "Credit Card Fraud Detection", Machine Learning Group of the Université Libre de Bruxelles, Kaggle
  dataset `mlg-ulb/creditcardfraud`, file `creditcard.csv`, renamed here `ulb_creditcard.csv` and otherwise unchanged.
  Transactions made by European cardholders in two days of September 2013. The variables V1–V28 are principal
  components released with the data, and `Time` counts the seconds from the start of the collection.
- **Collected by:** the research collaboration of Worldline and the Machine Learning Group of ULB on big data mining
  and fraud detection.
- **Licence:** this database is made available under the Open Database License,
  https://opendatacommons.org/licenses/odbl/1-0/. Any rights in individual contents of the database are licensed under
  the Database Contents License, https://opendatacommons.org/licenses/dbcl/1-0/.
- **Citation:** A. Dal Pozzolo, O. Caelen, R. A. Johnson and G. Bontempi, "Calibrating probability with undersampling
  for unbalanced classification", IEEE Symposium Series on Computational Intelligence, 2015.

## carclaims: vehicle insurance claims

- **Source:** "Vehicle Insurance Claim Fraud Detection", Kaggle dataset `shivamb/vehicle-claim-fraud-detection`,
  file `fraud_oracle.csv`. Claims from 1994 to 1996, originally distributed with the Angoss KnowledgeSeeker software.
- **Licence:** CC0 1.0 (public domain), as declared on Kaggle by the uploader.
- **Citation:** C. Phua, D. Alahakoon and V. Lee, "Minority report in fraud detection: classification of skewed
  data", ACM SIGKDD Explorations Newsletter 6(1), 50–59, 2004.
- **Line endings:** the file on Kaggle has Windows line endings (CRLF, 3,618,564 bytes). The file here has the same
  content with Unix line endings (LF, 3,603,144 bytes), and the SHA-256 above refers to it. To start from the Kaggle
  file, convert it first, for example with `tr -d '\r' < fraud_oracle_kaggle.csv > data/fraud_oracle.csv`.
