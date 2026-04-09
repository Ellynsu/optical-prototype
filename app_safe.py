import streamlit as st
import tempfile
from pathlib import Path

from optical_database_cleaner import (
    load_raw_csv,
    standardize_dataframe,
    build_recall_list,
    build_summary,
)

st.set_page_config(page_title="Optical Client Recovery Prototype", page_icon="👓", layout="wide")

st.title("Optical Client Recovery Prototype")
st.success("Cleaner module loaded.")

st.subheader("Upload Customer Database")
uploaded_file = st.file_uploader("Drag and drop or browse for your CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir) / uploaded_file.name
            tmp_path.write_bytes(uploaded_file.getvalue())

            raw_df = load_raw_csv(tmp_path)
            cleaned_df = standardize_dataframe(raw_df)
            recall_df = build_recall_list(cleaned_df)
            summary = build_summary(cleaned_df)

        st.subheader("Business Overview")

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Customers", summary.get("total_records", 0))
        c2.metric("Contactable", summary.get("contactable_records", 0))
        c3.metric("Marketing Eligible", summary.get("marketing_eligible_records", 0))
        c4.metric("Inactive (12+ mo)", summary.get("inactive_records_12m_plus", 0))
        c5.metric("High Priority Recall", summary.get("high_priority_recall", 0))

        st.info(
            f"Estimated immediate recall opportunity: {summary.get('high_priority_recall', 0)} customers ready for re-engagement."
        )

        # Add explanation column
        def build_reason(row):
            reasons = []

            days_purchase = row.get("days_since_last_purchase")
            if days_purchase is not None and str(days_purchase) != "nan":
                if days_purchase >= 365:
                    reasons.append(f"inactive {int(days_purchase)} days")

            rx_days = row.get("days_until_prescription_expiration")
            if rx_days is not None and str(rx_days) != "nan":
                if rx_days < 0:
                    reasons.append("prescription expired")
                elif rx_days <= 60:
                    reasons.append("prescription expiring soon")

            ins_days = row.get("days_until_insurance_renewal")
            if ins_days is not None and str(ins_days) != "nan" and ins_days <= 90 and ins_days >= 0:
                reasons.append("insurance renewing soon")

            amount = row.get("last_purchase_amount_clean")
            if amount is not None and str(amount) != "nan" and amount >= 300:
                reasons.append("higher prior spend")

            channel = row.get("recommended_channel")
            if channel:
                reasons.append(f"reachable by {channel}")

            return ", ".join(reasons) if reasons else "general recovery fit"

        if not recall_df.empty:
            recall_df = recall_df.copy()
            recall_df["why_this_customer"] = recall_df.apply(build_reason, axis=1)

        st.subheader("Recall Filters")

        f1, f2, f3 = st.columns(3)

        with f1:
            priority_options = sorted(recall_df["recovery_priority"].dropna().unique().tolist()) if not recall_df.empty and "recovery_priority" in recall_df.columns else []
            selected_priorities = st.multiselect(
                "Priority",
                options=priority_options,
                default=priority_options,
            )

        with f2:
            channel_options = sorted(recall_df["recommended_channel"].dropna().unique().tolist()) if not recall_df.empty and "recommended_channel" in recall_df.columns else []
            selected_channels = st.multiselect(
                "Recommended Channel",
                options=channel_options,
                default=channel_options,
            )

        with f3:
            min_score = st.slider("Minimum Recovery Score", 0, 100, 45, 5)

        filtered_recall_df = recall_df.copy()

        if not filtered_recall_df.empty:
            if selected_priorities:
                filtered_recall_df = filtered_recall_df[
                    filtered_recall_df["recovery_priority"].isin(selected_priorities)
                ]

            if selected_channels:
                filtered_recall_df = filtered_recall_df[
                    filtered_recall_df["recommended_channel"].isin(selected_channels)
                ]

            if "recovery_score" in filtered_recall_df.columns:
                filtered_recall_df = filtered_recall_df[
                    filtered_recall_df["recovery_score"] >= min_score
                ]

        st.subheader("Top Recall Opportunities")
        st.caption("Recovery score is based on inactivity, prescription timing, insurance timing, and prior purchase behavior.")

        preview_cols = [
            "first_name",
            "last_name",
            "email_clean",
            "phone_formatted",
            "recommended_channel",
            "days_since_last_purchase",
            "recovery_score",
            "recovery_priority",
            "why_this_customer",
        ]
        preview_cols = [c for c in preview_cols if c in filtered_recall_df.columns]

        if filtered_recall_df.empty:
            st.warning("No recall candidates match the current filters.")
        else:
            st.dataframe(
                filtered_recall_df[preview_cols].sort_values("recovery_score", ascending=False),
                use_container_width=True,
                hide_index=True,
            )

        st.subheader("Export Recall List")
        st.download_button(
            "Download filtered recall list as CSV",
            data=filtered_recall_df.to_csv(index=False).encode("utf-8"),
            file_name="optical_recall_list_filtered.csv",
            mime="text/csv",
            use_container_width=True,
        )

    except Exception as e:
        st.error("Processing failed.")
        st.exception(e)