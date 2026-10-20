# Data Card: Olist Brazilian E-Commerce Dataset (v2 Research Contract)

> Corrected 2026-10-20 against the raw files: status breakdown of excluded orders, out-of-bounds geolocation counts and the valid-row count. Every count below is reproduced by `scripts/report_analysis.py` (facts `raw.*`, `pop.*`, `dq.*`).

## 1. Raw Dataset Inventory & Cryptographic Fingerprints

All 9 source files reside in `data/` and are verified read-only via SHA-256 digests prior to every pipeline run:

| File Name | Rows | Primary / Grain Key | Role in Pipeline | SHA-256 Digest |
| :--- | ---: | :--- | :--- | :--- |
| `olist_orders_dataset.csv` | 99,441 | `order_id` | Master order lifecycle timestamps & status | `8df58ef3d2d7e9944010f7beecd9b75367f5588ec6e3c91cec19ae3345ef9ecf` |
| `olist_order_items_dataset.csv` | 112,650 | `(order_id, order_item_id)` | Item prices, freight, `product_id`, `seller_id` | `0bc4d068c4fe38cbb01bd90e8746e3c613fe7b4baef75fab7b0e329701c3e279` |
| `olist_order_payments_dataset.csv` | 103,886 | `(order_id, payment_sequential)` | Payment value, installments, type (`extended_purchase_assumptions_v2` ablation only) | `4f713964f2815dbbaa40b9488268c55aac3627bfce5aa96cf58d1f3616de3cc0` |
| `olist_order_reviews_dataset.csv` | 99,224 | `(review_id, order_id)` | Post-delivery customer reviews (**blocked from model features**; used only for business motivation) | `012b61c7593e34f51fa614efdf802b9c7056ce6aae5307ddb93236e7cfc797d7` |
| `olist_products_dataset.csv` | 32,951 | `product_id` | Product weight, dimensions, description length, category | `3e6569628a17fbc75fd206ee357b59e20364b9afa90f5b6cd5b4d624c58aa9cc` |
| `olist_customers_dataset.csv` | 99,441 | `customer_id` (`customer_unique_id`) | Customer ZIP prefix, city, state (`customer_state`) | `983a422239e1712ded753b3bf9ecf47dc73f144d306029dcfa99e70a226883d2` |
| `olist_sellers_dataset.csv` | 3,095 | `seller_id` | Seller ZIP prefix, city, state (`seller_state`) | `1f643d2b950373b85735e7794b20986f528d7a000432e7c6f9bcbb44d0846a0e` |
| `olist_geolocation_dataset.csv` | 1,000,163 | `geolocation_zip_code_prefix` (non-unique) | Lat/Lng coordinate samples per 5-digit ZIP prefix | `b514f6fc991b9566aeba02aa5d67e2c3630f034b60a0e05aa0d082a3b66d88d6` |
| `product_category_name_translation.csv` | 71 | `product_category_name` | Portuguese-to-English category translation lookup | `a81f0d1f27b27e7293f761bc79e3ce8f348ee39c4b3ed3e49bde38f478586278` |

---

## 2. Exact Order Cohort Reconciliation (`99,441 = 96,470 + 2,971`)

Every source order in `olist_orders_dataset.csv` is accounted for without silent row drops:

| Cohort / Segment | Order Count | Share of Total (`99,441`) | Target (`late`) Value | Operational & Modeling Treatment |
| :--- | ---: | ---: | :--- | :--- |
| **Total Source Orders (Path B)** | **99,441** | **100.00%** | Nullable `Int8` | Full operational purchase stream |
| **Cohort A (Path A Evaluable)** | **96,470** | **97.01%** | `0` (`89,936`) or `1` (`6,534`) | `order_status == "delivered"`, `has_items == True`, non-null delivery & estimated dates |
| **Excluded / Unknown Outcome Orders** | **2,971** | **2.99%** | **`pd.NA` (Never `0`)** | Retained in Path B operational denominator; excluded from supervised loss |
| ├─ Excluded without item rows (`has_items == False`) | 775 | 0.78% | `pd.NA` | Unavailable (`603`), canceled (`164`), created (`5`), invoiced (`2`), shipped (`1`); scored via `fallback_no_items` prior |
| └─ Excluded with item rows (`has_items == True`, non-delivered / missing ts) | 2,196 | 2.21% | `pd.NA` | Shipped (`1,106`), canceled (`461`), invoiced (`312`), processing (`301`), delivered missing delivery ts (`8`), unavailable (`6`), approved (`2`) |

---

## 3. Data Quality Anomalies, Limitations & Mitigations

1. **Midnight Estimated Delivery Date Storage (`00:00:00`)**:
   - `order_estimated_delivery_date` is stored as a midnight timestamp (`YYYY-MM-DD 00:00:00`), whereas `order_delivered_customer_date` includes exact time-of-day (`HH:MM:SS`).
   - Naive timestamp comparison (`delivered > estimated`) marks `1,292` orders delivered on the promised calendar day between `00:00:01` and `23:59:59` as "late".
   - **Mitigation**: Primary label `late_calendar_day` compares calendar dates (`dt.floor("D")`), yielding `6,534` late orders (`6.77%`), while `late_timestamp` (`7,826` late orders, `8.11%`) is preserved as an explicit audit sensitivity analysis.
2. **Out-of-Bounds Geolocation Coordinates (`31` raw rows, `27` after de-duplication)**:
   - `olist_geolocation_dataset.csv` contains `1,000,163` rows with `261,831` exact duplicate rows and `31` rows (`27` unique) falling outside Brazil's geographic bounding box (`lat in [-33.8, 5.4]`, `lng in [-74.1, -28.8]`, including coordinates in the Atlantic Ocean and Europe).
   - **Mitigation**: Deduplicate exact coordinate rows first (`738,332` unique rows), filter strictly to `[-33.8, 5.4] x [-74.1, -28.8]` (`738,305` valid rows across `19,011` ZIP prefixes), compute median centroids per ZIP prefix, and apply an explicit fallback hierarchy (`zip_prefix` $\to$ `state_centroid` $\to$ `national_centroid` $\to$ `missing`). If no valid coordinates exist in a synthetic/corrupt fixture, return `NaN` rather than averaging invalid coordinates.
3. **Multi-Seller Order Structure (`1,278` multi-seller orders)**:
   - While `97,388` orders with items involve a single seller, `1,278` orders contain items fulfilled by 2 to 5 distinct sellers (`100,010` unique `(order_id, seller_id)` pairs across `112,650` item lines).
   - **Mitigation**: Seller history features are computed on the deduplicated `(order_id, seller_id)` bridge table and aggregated to order level via `mean`. Route features (and therefore the selected `rule_region_lead`) use the primary seller only: the item with the highest price, then freight, then lowest item id.
4. **Right-Censoring & Maturity Selection Bias in Late Validation Folds**:
   - Because long-distance and severely delayed orders take weeks to deliver, truncating a validation fold at a fixed decision date (`selection_at = 2018-05-26 18:18:03`) disproportionately censors late deliveries in the most recent fold (`val_2018_04`).
   - Specifically, in `2018-04`, `104` Cohort A orders were delivered after `2018-05-26 18:18:03`, and `89` of those `104` (`85.58%`) were late.
   - **Mitigation**: `cohort-observation-analysis.csv` quantifies monthly censoring rates, and validation reports compare selection-time matured metrics against post-maturity metrics.
5. **Payment Timestamp Ambiguity**:
   - `olist_order_payments_dataset.csv` does not record when each payment installment or voucher was authorized relative to `order_purchase_timestamp`.
   - **Mitigation**: Payment features are excluded from the primary `base_v2` feature set and tested only in `extended_purchase_assumptions_v2`.
6. **Timezone Non-Explicitness**:
   - All timestamps in Olist are timezone-naive local timestamps (predominantly Brasília Time, `BRT / UTC-3`, with minor Western state offsets not encoded in the source CSVs). All daily cutoffs (`23:59:59.999999`) are defined in source timestamp units.
