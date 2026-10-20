"""Shared pytest fixtures for Olist Delivery Delay Warning verification suite (T01-T18)."""
from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = PROJECT_ROOT / "src"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))


@pytest.fixture(scope="session")
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture
def synthetic_label_orders() -> pd.DataFrame:
    """Hand-calculated edge cases for T01 (label boundary) and T02 (nullable unknown outcomes)."""
    return pd.DataFrame(
        [
            # Case 0: Early delivery (1 day before estimated date) -> both 0
            {
                "order_id": "ord_early",
                "customer_id": "c0",
                "order_status": "delivered",
                "has_items": True,
                "order_purchase_timestamp": "2018-03-01 10:00:00",
                "order_approved_at": "2018-03-01 10:15:00",
                "order_delivered_carrier_date": "2018-03-02 10:00:00",
                "order_delivered_customer_date": "2018-03-09 14:30:00",
                "order_estimated_delivery_date": "2018-03-10 00:00:00",
            },
            # Case 1: Same-day 23:59:59 on estimated date -> calendar_day=0, timestamp=1, same_day_after_midnight=True
            {
                "order_id": "ord_same_day_late_ts",
                "customer_id": "c1",
                "order_status": "delivered",
                "has_items": True,
                "order_purchase_timestamp": "2018-03-01 11:00:00",
                "order_approved_at": "2018-03-01 11:15:00",
                "order_delivered_carrier_date": "2018-03-02 11:00:00",
                "order_delivered_customer_date": "2018-03-10 23:59:59",
                "order_estimated_delivery_date": "2018-03-10 00:00:00",
            },
            # Case 2: Exact midnight on estimated date -> calendar_day=0, timestamp=0
            {
                "order_id": "ord_exact_midnight",
                "customer_id": "c2",
                "order_status": "delivered",
                "has_items": True,
                "order_purchase_timestamp": "2018-03-01 12:00:00",
                "order_approved_at": "2018-03-01 12:15:00",
                "order_delivered_carrier_date": "2018-03-02 12:00:00",
                "order_delivered_customer_date": "2018-03-10 00:00:00",
                "order_estimated_delivery_date": "2018-03-10 00:00:00",
            },
            # Case 3: Next-day 00:00:00 -> calendar_day=1, timestamp=1
            {
                "order_id": "ord_next_day_midnight",
                "customer_id": "c3",
                "order_status": "delivered",
                "has_items": True,
                "order_purchase_timestamp": "2018-03-01 13:00:00",
                "order_approved_at": "2018-03-01 13:15:00",
                "order_delivered_carrier_date": "2018-03-02 13:00:00",
                "order_delivered_customer_date": "2018-03-11 00:00:00",
                "order_estimated_delivery_date": "2018-03-10 00:00:00",
            },
            # Case 4: Canceled order without delivery date -> pd.NA (never 0!)
            {
                "order_id": "ord_canceled",
                "customer_id": "c4",
                "order_status": "canceled",
                "has_items": True,
                "order_purchase_timestamp": "2018-03-02 09:00:00",
                "order_approved_at": "2018-03-02 09:15:00",
                "order_delivered_carrier_date": None,
                "order_delivered_customer_date": None,
                "order_estimated_delivery_date": "2018-03-15 00:00:00",
            },
            # Case 5: Shipped / in-transit without delivery date -> pd.NA (never 0!)
            {
                "order_id": "ord_shipped",
                "customer_id": "c5",
                "order_status": "shipped",
                "has_items": True,
                "order_purchase_timestamp": "2018-03-02 10:00:00",
                "order_approved_at": "2018-03-02 10:15:00",
                "order_delivered_carrier_date": "2018-03-03 10:00:00",
                "order_delivered_customer_date": None,
                "order_estimated_delivery_date": "2018-03-15 00:00:00",
            },
            # Case 6: Delivered status but missing delivery date -> pd.NA
            {
                "order_id": "ord_delivered_missing_ts",
                "customer_id": "c6",
                "order_status": "delivered",
                "has_items": True,
                "order_purchase_timestamp": "2018-03-02 11:00:00",
                "order_approved_at": "2018-03-02 11:15:00",
                "order_delivered_carrier_date": "2018-03-03 11:00:00",
                "order_delivered_customer_date": None,
                "order_estimated_delivery_date": "2018-03-15 00:00:00",
            },
            # Case 7: Unavailable status without delivery date -> pd.NA
            {
                "order_id": "ord_unavailable",
                "customer_id": "c7",
                "order_status": "unavailable",
                "has_items": False,
                "order_purchase_timestamp": "2018-03-02 12:00:00",
                "order_approved_at": "2018-03-02 12:15:00",
                "order_delivered_carrier_date": None,
                "order_delivered_customer_date": None,
                "order_estimated_delivery_date": "2018-03-15 00:00:00",
            },
        ]
    )
