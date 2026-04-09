from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


TODAY = pd.Timestamp.today().normalize()
RECALL_LOOKBACK_DAYS = 365
HIGH_VALUE_THRESHOLD = 300.0
INSURANCE_WINDOW_DAYS = 90
PRESCRIPTION_WINDOW_DAYS = 60


def normalize_column_name(name: str) -> str:
    name = str(name).strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return name.strip("_")


def clean_text(value) -> Optional[str]:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "none", "null"}:
        return None
    return text


def clean_email(value) -> Optional[str]:
    text = clean_text(value)
    if not text:
        return None
    text = text.lower().strip()
    text = re.sub(r"\s+", "", text)
    if not re.match(r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$", text):
        return None
    return text


def clean_phone(value) -> tuple[Optional[str], Optional[str]]:
    text = clean_text(value)
    if not text:
        return None, None

    digits = re.sub(r"\D", "", text)
    if len(digits) == 11 and digits.startswith("1"):
        digits = digits[1:]
    if len(digits) != 10:
        return None, None

    formatted = f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"
    return digits, formatted


def parse_currency(value) -> Optional[float]:
    text = clean_text(value)
    if not text:
        return np.nan
    text = re.sub(r"[^0-9.\-]", "", text)
    if text in {"", ".", "-", "-."}:
        return np.nan
    try:
        return float(text)
    except ValueError:
        return np.nan


def parse_date(value) -> pd.Timestamp:
    text = clean_text(value)
    if not text:
        return pd.NaT
    return pd.to_datetime(text, errors="coerce")


def normalize_yes_no(value) -> Optional[bool]:
    text = clean_text(value)
    if not text:
        return None
    text = text.lower()
    if text in {"yes", "y", "true", "1"}:
        return True
    if text in {"no", "n", "false", "0"}:
        return False
    return None


def normalize_contact_method(value) -> Optional[str]:
    text = clean_text(value)
    if not text:
        return None
    text = text.lower()
    if text in {"email", "e-mail"}:
        return "email"
    if text in {"sms", "text", "text_message"}:
        return "sms"
    if text in {"phone", "call"}:
        return "phone"
    return text


def days_since(series: pd.Series) -> pd.Series:
    return (TODAY - series).dt.days


def days_until(series: pd.Series) -> pd.Series:
    return (series - TODAY).dt.days


def find_header_row(raw: pd.DataFrame) -> int:
    for i in range(len(raw)):
        row_values = [str(v).strip().lower() for v in raw.iloc[i].tolist()]
        if "customer_id" in row_values and "first_name" in row_values:
            return i
    raise ValueError("Could not find the real header row in the CSV.")


def segment_inactivity(days_value: float) -> str:
    if pd.isna(days_value):
        return "unknown"
    if days_value < 365:
        return "active_lt_12m"
    if days_value < 730:
        return "inactive_12_24m"
    if days_value < 1095:
        return "inactive_24_36m"
    return "inactive_36m_plus"


def load_raw_csv(input_path: Path | str) -> pd.DataFrame:
    raw = pd.read_csv(input_path, header=None)
    header_row = find_header_row(raw)

    headers = [normalize_column_name(x) for x in raw.iloc[header_row].tolist()]
    data = raw.iloc[header_row + 1 :].copy()
    data.columns = headers
    data = data.reset_index(drop=True)
    data = data.dropna(how="all").reset_index(drop=True)
    return data


def standardize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].map(clean_text)

    if "email" in df.columns:
        df["email_clean"] = df["email"].map(clean_email)

    if "phone" in df.columns:
        phone_clean = df["phone"].map(clean_phone)
        df["phone_digits"] = [x[0] for x in phone_clean]
        df["phone_formatted"] = [x[1] for x in phone_clean]

    if "preferred_contact_method" in df.columns:
        df["preferred_contact_method"] = df["preferred_contact_method"].map(normalize_contact_method)

    for col in [
        "last_eye_exam_date",
        "prescription_expiration_date",
        "last_purchase_date",
        "insurance_renewal_date",
        "last_reminder_sent_date",
    ]:
        if col in df.columns:
            df[col] = df[col].map(parse_date)

    if "last_purchase_amount" in df.columns:
        df["last_purchase_amount_clean"] = df["last_purchase_amount"].map(parse_currency)

    for col in ["has_vision_insurance", "consent_to_marketing"]:
        if col in df.columns:
            df[col + "_bool"] = df[col].map(normalize_yes_no)

    for col in ["first_name", "last_name"]:
        if col in df.columns:
            df[col] = df[col].str.title()

    if "last_purchase_date" in df.columns:
        df["days_since_last_purchase"] = days_since(df["last_purchase_date"])
        df["inactivity_segment"] = df["days_since_last_purchase"].map(segment_inactivity)

    if "last_eye_exam_date" in df.columns:
        df["days_since_last_eye_exam"] = days_since(df["last_eye_exam_date"])

    if "last_reminder_sent_date" in df.columns:
        df["days_since_last_reminder"] = days_since(df["last_reminder_sent_date"])

    if "prescription_expiration_date" in df.columns:
        df["days_until_prescription_expiration"] = days_until(df["prescription_expiration_date"])
        df["prescription_expired"] = df["days_until_prescription_expiration"] < 0
        df["prescription_expiring_soon"] = df["days_until_prescription_expiration"].between(0, PRESCRIPTION_WINDOW_DAYS)

    if "insurance_renewal_date" in df.columns:
        df["days_until_insurance_renewal"] = days_until(df["insurance_renewal_date"])
        df["insurance_renewing_soon"] = df["days_until_insurance_renewal"].between(0, INSURANCE_WINDOW_DAYS)

    df["has_valid_email"] = df.get("email_clean", pd.Series(index=df.index)).notna()
    df["has_valid_phone"] = df.get("phone_digits", pd.Series(index=df.index)).notna()
    df["is_contactable"] = df["has_valid_email"] | df["has_valid_phone"]
    df["marketing_eligible"] = df.get("consent_to_marketing_bool", pd.Series(False, index=df.index)).fillna(False) & df["is_contactable"]

    df["inactive_customer"] = df.get("days_since_last_purchase", pd.Series(np.nan, index=df.index)) >= RECALL_LOOKBACK_DAYS
    df["high_value_customer"] = df.get("last_purchase_amount_clean", pd.Series(np.nan, index=df.index)) >= HIGH_VALUE_THRESHOLD

    def recommend_channel(row) -> Optional[str]:
        pref = row.get("preferred_contact_method")
        has_email = bool(row.get("has_valid_email"))
        has_phone = bool(row.get("has_valid_phone"))

        if pref == "email" and has_email:
            return "email"
        if pref in {"sms", "phone"} and has_phone:
            return pref
        if has_email:
            return "email"
        if has_phone:
            return "sms"
        return None

    df["recommended_channel"] = df.apply(recommend_channel, axis=1)

    score = np.zeros(len(df), dtype=float)

    days_purchase = df.get("days_since_last_purchase", pd.Series(np.nan, index=df.index))
    score += np.where(days_purchase.between(365, 730, inclusive="both"), 30, 0)
    score += np.where(days_purchase.between(731, 1095, inclusive="both"), 20, 0)
    score += np.where(days_purchase > 1095, 8, 0)
    score += np.where(days_purchase < 365, -15, 0)

    days_exam = df.get("days_since_last_eye_exam", pd.Series(np.nan, index=df.index))
    score += np.where(days_exam >= 365, 15, 0)
    score += np.where(days_exam >= 730, 10, 0)

    score += np.where(df.get("prescription_expiring_soon", pd.Series(False, index=df.index)).fillna(False), 18, 0)
    score += np.where(df.get("prescription_expired", pd.Series(False, index=df.index)).fillna(False), 12, 0)
    score += np.where(df.get("insurance_renewing_soon", pd.Series(False, index=df.index)).fillna(False), 20, 0)

    purchase_amount = df.get("last_purchase_amount_clean", pd.Series(np.nan, index=df.index)).fillna(0)
    score += np.where(purchase_amount >= 500, 12, 0)
    score += np.where((purchase_amount >= 250) & (purchase_amount < 500), 6, 0)

    score += np.where(df["marketing_eligible"], 15, 0)
    score += np.where(df["is_contactable"], 8, 0)
    score += np.where(~df["is_contactable"], -40, 0)

    days_reminder = df.get("days_since_last_reminder", pd.Series(np.nan, index=df.index))
    score += np.where(days_reminder.between(0, 30, inclusive="both"), -10, 0)

    df["recovery_score"] = np.clip(score, 0, 100).round(0).astype(int)

    def bucket(score_value: int) -> str:
        if score_value >= 70:
            return "high"
        if score_value >= 45:
            return "medium"
        return "low"

    df["recovery_priority"] = df["recovery_score"].map(bucket)

    return df


def build_recall_list(df: pd.DataFrame) -> pd.DataFrame:
    recall = df.copy()

    recall = recall[
        (recall["inactive_customer"] | recall.get("prescription_expiring_soon", False) | recall.get("insurance_renewing_soon", False))
        & (recall["marketing_eligible"])
    ].copy()

    recall = recall.sort_values(
        by=["recovery_score", "days_since_last_purchase", "last_purchase_amount_clean"],
        ascending=[False, False, False],
        na_position="last",
    )

    keep_cols = [
        "customer_id",
        "first_name",
        "last_name",
        "email_clean",
        "phone_formatted",
        "preferred_contact_method",
        "recommended_channel",
        "last_purchase_date",
        "days_since_last_purchase",
        "inactivity_segment",
        "last_purchase_category",
        "last_purchase_amount_clean",
        "last_eye_exam_date",
        "days_since_last_eye_exam",
        "prescription_expiration_date",
        "days_until_prescription_expiration",
        "insurance_provider",
        "insurance_renewal_date",
        "days_until_insurance_renewal",
        "recovery_score",
        "recovery_priority",
        "notes",
    ]

    existing_cols = [c for c in keep_cols if c in recall.columns]
    recall = recall[existing_cols]
    return recall


def build_summary(df: pd.DataFrame) -> dict:
    return {
        "total_records": int(len(df)),
        "contactable_records": int(df["is_contactable"].sum()),
        "marketing_eligible_records": int(df["marketing_eligible"].sum()),
        "inactive_records_12m_plus": int(df["inactive_customer"].sum()),
        "high_priority_recall": int((df["recovery_priority"] == "high").sum()),
        "medium_priority_recall": int((df["recovery_priority"] == "medium").sum()),
        "low_priority_recall": int((df["recovery_priority"] == "low").sum()),
        "missing_valid_email": int((~df["has_valid_email"]).sum()),
        "missing_valid_phone": int((~df["has_valid_phone"]).sum()),
        "top_inactivity_segments": df["inactivity_segment"].value_counts(dropna=False).to_dict(),
        "recommended_channels": df["recommended_channel"].value_counts(dropna=False).to_dict(),
    }


def run_pipeline(input_csv: Path | str, output_dir: Path | str):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_df = load_raw_csv(input_csv)
    cleaned_df = standardize_dataframe(raw_df)
    recall_df = build_recall_list(cleaned_df)
    summary = build_summary(cleaned_df)

    cleaned_path = output_dir / "optical_clients_cleaned.csv"
    recall_path = output_dir / "optical_clients_recall_candidates.csv"
    summary_path = output_dir / "optical_clients_summary.json"

    cleaned_df.to_csv(cleaned_path, index=False)
    recall_df.to_csv(recall_path, index=False)

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    return {
        "cleaned_df": cleaned_df,
        "recall_df": recall_df,
        "summary": summary,
        "cleaned_path": cleaned_path,
        "recall_path": recall_path,
        "summary_path": summary_path,
    }


if __name__ == "__main__":
    input_file = "optical_client_sample_data - Raw_Data.csv"
    output_dir = "output"
    result = run_pipeline(input_file, output_dir)
    print("Done.")
    print(f"Cleaned file: {result['cleaned_path']}")
    print(f"Recall candidates: {result['recall_path']}")
    print(f"Summary: {result['summary_path']}")
