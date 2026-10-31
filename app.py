
import io
import math
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image as RLImage, PageBreak
)

st.set_page_config(page_title="SGS Analysis Software", page_icon="📈", layout="wide")

REQUIRED = ["Station_ID", "Time_min", "Pressure_psi", "TVD_ft"]
OPTIONAL = [
    "Temperature_C", "Station_Start_Time_min", "Station_End_Time_min",
    "Station_Measurement_Time_min", "Correction_Duration_min",
    "WHSIP_psi_at_TimeZero", "WHSIP_TimeZero_min"
]

def read_input(uploaded):
    name = uploaded.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded)
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded)
    raise ValueError("Please upload a CSV or Excel file.")

def numeric_clean(df):
    df = df.copy()
    for col in ["Time_min", "Pressure_psi", "TVD_ft"] + [c for c in OPTIONAL if c in df.columns and c != "WHSIP_psi_at_TimeZero"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def analyze_sgs(raw, correction_factor=1.0, whsip_override=None):
    missing = [c for c in REQUIRED if c not in raw.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))

    data = numeric_clean(raw)
    data["Station_ID"] = data["Station_ID"].astype(str).str.strip()
    data = data.dropna(subset=["Station_ID", "Time_min", "Pressure_psi", "TVD_ft"])
    if data.empty:
        raise ValueError("No valid rows remain after removing blank/non-numeric required values.")

    # Preserve chronological order and require unique station measurement times.
    data = data.sort_values(["Time_min"]).reset_index(drop=True)
    station_rows = []
    station_ids = data["Station_ID"].drop_duplicates().tolist()

    for sid in station_ids:
        g = data[data["Station_ID"] == sid].sort_values("Time_min").copy()
        if len(g) < 3:
            raise ValueError(f"{sid} has only {len(g)} usable readings. At least 3 are required for a meaningful regression.")
        if g["Time_min"].nunique() < 2:
            raise ValueError(f"{sid} must contain at least two distinct timestamps.")

        # Station boundaries: use supplied values if available, otherwise sample min/max.
        start = float(g["Station_Start_Time_min"].dropna().iloc[0]) if "Station_Start_Time_min" in g and g["Station_Start_Time_min"].notna().any() else float(g["Time_min"].min())
        end = float(g["Station_End_Time_min"].dropna().iloc[0]) if "Station_End_Time_min" in g and g["Station_End_Time_min"].notna().any() else float(g["Time_min"].max())
        if end <= start:
            raise ValueError(f"{sid}: station end time must be greater than start time.")

        # Fit only readings inside the declared station recording window.
        fit = g[(g["Time_min"] >= start - 1e-9) & (g["Time_min"] <= end + 1e-9)]
        if len(fit) < 3:
            raise ValueError(f"{sid}: fewer than 3 pressure samples fall inside its station start/end window.")

        x = fit["Time_min"].to_numpy(dtype=float)
        y = fit["Pressure_psi"].to_numpy(dtype=float)
        slope, intercept = np.polyfit(x, y, 1)
        pred = slope * x + intercept
        ss_res = float(np.sum((y - pred) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 1.0 if ss_tot == 0 and ss_res == 0 else (0.0 if ss_tot == 0 else 1 - ss_res / ss_tot)

        measurement_time = (
            float(g["Station_Measurement_Time_min"].dropna().iloc[0])
            if "Station_Measurement_Time_min" in g and g["Station_Measurement_Time_min"].notna().any()
            else end
        )
        eligible = g[g["Time_min"] <= measurement_time + 1e-9]
        if eligible.empty:
            raise ValueError(f"{sid}: no pressure sample exists at or before its measurement time.")
        measured_pressure = float(eligible.sort_values("Time_min").iloc[-1]["Pressure_psi"])
        tvd = float(g["TVD_ft"].dropna().iloc[0])

        duration = (
            float(g["Correction_Duration_min"].dropna().iloc[0])
            if "Correction_Duration_min" in g and g["Correction_Duration_min"].notna().any()
            else np.nan
        )

        station_rows.append({
            "Station_ID": sid,
            "TVD_ft": tvd,
            "Station_Start_Time_min": start,
            "Station_End_Time_min": end,
            "Measurement_Time_min": measurement_time,
            "Measured_Pressure_psi": measured_pressure,
            "Buildup_Rate_psi_min": float(slope),
            "Regression_Intercept": float(intercept),
            "R_squared": float(r2),
            "Regression_Sample_Count": int(len(fit)),
            "Correction_Duration_min": duration,
            "Temperature_C": float(g["Temperature_C"].dropna().mean()) if "Temperature_C" in g and g["Temperature_C"].notna().any() else np.nan
        })

    result = pd.DataFrame(station_rows).sort_values("Measurement_Time_min").reset_index(drop=True)
    if result["TVD_ft"].duplicated().any():
        raise ValueError("Each station must have a unique TVD for station-to-station gradient calculations.")
    if result["Measurement_Time_min"].duplicated().any():
        raise ValueError("Station measurement times must be unique and chronological.")

    # Time-zero is the start of the first station's recording.
    time_zero = float(result.iloc[0]["Station_Start_Time_min"])

    # Use supplied durations where present. Otherwise infer elapsed time since
    # the preceding station measurement, with the first interval from time-zero.
    durations = []
    previous_time = time_zero
    for _, row in result.iterrows():
        inferred = float(row["Measurement_Time_min"] - previous_time)
        supplied = row["Correction_Duration_min"]
        duration = float(supplied) if pd.notna(supplied) else inferred
        if duration < 0:
            raise ValueError("Correction durations cannot be negative.")
        # A mismatch is retained as a visible QC warning rather than silently altered.
        durations.append(duration)
        previous_time = float(row["Measurement_Time_min"])

    result["Correction_Duration_min"] = durations
    result["Applied_Buildup_Rate_psi_min"] = result["Buildup_Rate_psi_min"] * float(correction_factor)
    result["Pressure_Contribution_psi"] = result["Applied_Buildup_Rate_psi_min"] * result["Correction_Duration_min"]
    result["Cumulative_Correction_psi"] = result["Pressure_Contribution_psi"].cumsum()
    result["Corrected_Pressure_psi"] = result["Measured_Pressure_psi"] - result["Cumulative_Correction_psi"]

    result = result.sort_values("TVD_ft").reset_index(drop=True)
    result["Raw_Gradient_psi_ft"] = result["Measured_Pressure_psi"].diff() / result["TVD_ft"].diff()
    result["Corrected_Gradient_psi_ft"] = result["Corrected_Pressure_psi"].diff() / result["TVD_ft"].diff()

    # Optional soft QC: linear extrapolation of corrected pressure-depth trend to TVD=0.
    whsip = whsip_override
    if whsip is None and "WHSIP_psi_at_TimeZero" in data.columns:
        vals = pd.to_numeric(data["WHSIP_psi_at_TimeZero"], errors="coerce").dropna()
        if not vals.empty:
            whsip = float(vals.iloc[0])

    if len(result) >= 2:
        depth_slope, depth_intercept = np.polyfit(result["TVD_ft"], result["Corrected_Pressure_psi"], 1)
        extrapolated_whp = float(depth_intercept)
    else:
        extrapolated_whp = np.nan

    qc_difference = (extrapolated_whp - float(whsip)) if whsip is not None and np.isfinite(extrapolated_whp) else np.nan

    # Regression diagnostics and duration consistency.
    warnings = []
    if (result["Buildup_Rate_psi_min"] < 0).any():
        warnings.append("One or more station regression slopes are negative. Check pressure-time data and station boundaries.")
    if (result["R_squared"] < 0.90).any():
        warnings.append("One or more station regressions have R² below 0.90. Inspect those pressure-time trends.")
    if not np.all(np.diff(result["TVD_ft"]) > 0):
        warnings.append("TVD values are not strictly increasing after sorting; verify survey order and depth values.")
    total_elapsed = float(result["Measurement_Time_min"].max() - time_zero)
    total_correction_time = float(result["Correction_Duration_min"].sum())
    if abs(total_elapsed - total_correction_time) > max(0.05, 0.01 * max(total_elapsed, 1)):
        warnings.append(
            f"Sum of correction durations ({total_correction_time:.2f} min) differs from elapsed time "
            f"from time-zero to last measurement ({total_elapsed:.2f} min). Verify movement/recording intervals "
            "to avoid gaps or double counting."
        )

    meta = {
        "time_zero_min": time_zero,
        "whsip_psi": whsip,
        "extrapolated_whsip_psi": extrapolated_whp,
        "whsip_difference_psi": qc_difference,
        "correction_factor": float(correction_factor),
        "station_count": int(len(result)),
        "raw_rows": int(len(data)),
        "total_elapsed_min": total_elapsed,
        "total_correction_time_min": total_correction_time,
        "warnings": warnings
    }
    return data, result, meta

def make_figures(data, result):
    figs = {}

    fig1, ax = plt.subplots(figsize=(8, 5))
    ax.plot(result["Measured_Pressure_psi"], result["TVD_ft"], marker="o", label="Raw pressure")
    ax.plot(result["Corrected_Pressure_psi"], result["TVD_ft"], marker="s", label="Time-zero corrected")
    ax.set_xlabel("Pressure (psi)")
    ax.set_ylabel("TVD (ft)")
    ax.set_title("Raw vs Corrected SGS Pressure Profile")
    ax.grid(True, alpha=0.3); ax.legend(); ax.invert_yaxis()
    fig1.tight_layout(); figs["profile"] = fig1

    fig2, ax = plt.subplots(figsize=(8, 5))
    for sid, g in data.groupby("Station_ID", sort=False):
        ax.plot(g["Time_min"], g["Pressure_psi"], marker=".", label=sid)
    ax.set_xlabel("Elapsed time (min)"); ax.set_ylabel("Pressure (psi)")
    ax.set_title("Downhole Pressure vs Time")
    ax.grid(True, alpha=0.3); ax.legend(ncol=2, fontsize=8)
    fig2.tight_layout(); figs["pressure_time"] = fig2

    fig3, ax = plt.subplots(figsize=(8, 5))
    ax.bar(result["Station_ID"], result["Buildup_Rate_psi_min"])
    ax.set_xlabel("Station"); ax.set_ylabel("Buildup rate (psi/min)")
    ax.set_title("Station-wise Transient Buildup Rates")
    ax.grid(True, axis="y", alpha=0.3)
    fig3.tight_layout(); figs["rates"] = fig3

    grad = result.dropna(subset=["Raw_Gradient_psi_ft", "Corrected_Gradient_psi_ft"])
    fig4, ax = plt.subplots(figsize=(8, 5))
    if not grad.empty:
        x = grad["TVD_ft"].iloc[1:] if len(grad) == len(result) else grad["TVD_ft"]
        ax.plot(grad["Raw_Gradient_psi_ft"], grad["TVD_ft"], marker="o", label="Raw gradient")
        ax.plot(grad["Corrected_Gradient_psi_ft"], grad["TVD_ft"], marker="s", label="Corrected gradient")
    ax.set_xlabel("Pressure gradient (psi/ft)"); ax.set_ylabel("TVD (ft)")
    ax.set_title("Raw vs Corrected Pressure Gradient")
    ax.grid(True, alpha=0.3); ax.legend(); ax.invert_yaxis()
    fig4.tight_layout(); figs["gradients"] = fig4

    return figs

def fig_png(fig):
    b = io.BytesIO()
    fig.savefig(b, format="png", dpi=170, bbox_inches="tight")
    b.seek(0)
    return b

def build_pdf(result, meta, figs, project_name, notes):
    buff = io.BytesIO()
    doc = SimpleDocTemplate(buff, pagesize=landscape(A4), rightMargin=14*mm, leftMargin=14*mm, topMargin=12*mm, bottomMargin=12*mm)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontSize=8, leading=10))
    story = []
    story.append(Paragraph("STATIC GRADIENT SURVEY (SGS) ANALYSIS REPORT", styles["Title"]))
    story.append(Paragraph(project_name.replace("&", "&amp;"), styles["Heading2"]))
    story.append(Spacer(1, 4*mm))
    summary = [
        ["Stations", str(meta["station_count"]), "Raw rows", str(meta["raw_rows"]), "Time zero (min)", f'{meta["time_zero_min"]:.2f}'],
        ["Correction factor", f'{meta["correction_factor"]:.3f}', "Elapsed time (min)", f'{meta["total_elapsed_min"]:.2f}', "Correction duration sum (min)", f'{meta["total_correction_time_min"]:.2f}']
    ]
    t = Table(summary, colWidths=[32*mm, 22*mm, 25*mm, 22*mm, 38*mm, 24*mm])
    t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),colors.whitesmoke),("GRID",(0,0),(-1,-1),0.4,colors.grey),("FONTSIZE",(0,0),(-1,-1),8),("PADDING",(0,0),(-1,-1),5)]))
    story.append(t); story.append(Spacer(1, 4*mm))

    story.append(Paragraph("Methodology", styles["Heading2"]))
    methodology = (
        "Station pressure-time samples are fitted with a least-squares straight line, P(t)=a+bt. "
        "The fitted slope b is the station buildup rate (psi/min). The transient pressure contribution "
        "for each station is calculated as adjusted buildup rate multiplied by its correction duration. "
        "These contributions are cumulatively summed and subtracted from the measured station pressure "
        "to estimate pressure referenced to the first station's recording start (time zero)."
    )
    story.append(Paragraph(methodology, styles["BodyText"]))
    story.append(Spacer(1, 3*mm))
    story.append(Paragraph("Station Results", styles["Heading2"]))

    cols = ["Station_ID","TVD_ft","Measurement_Time_min","Measured_Pressure_psi","Buildup_Rate_psi_min","R_squared","Correction_Duration_min","Cumulative_Correction_psi","Corrected_Pressure_psi","Corrected_Gradient_psi_ft"]
    display = result[cols].copy()
    display = display.replace({np.nan: ""})
    table_data = [[c.replace("_"," ") for c in cols]]
    for _, row in display.iterrows():
        vals=[]
        for c in cols:
            v=row[c]
            vals.append(f"{v:.4f}" if isinstance(v,(float,np.floating)) and v != "" else str(v))
        table_data.append(vals)
    widths=[18*mm,17*mm,24*mm,25*mm,24*mm,14*mm,25*mm,27*mm,25*mm,25*mm]
    tab=Table(table_data, colWidths=widths, repeatRows=1)
    tab.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#24476B")),
        ("TEXTCOLOR",(0,0),(-1,0),colors.white),
        ("GRID",(0,0),(-1,-1),0.3,colors.grey),
        ("FONTSIZE",(0,0),(-1,-1),6.3),
        ("PADDING",(0,0),(-1,-1),3),
        ("VALIGN",(0,0),(-1,-1),"MIDDLE")
    ]))
    story.append(tab)
    story.append(Spacer(1, 4*mm))

    story.append(Paragraph("Diagnostic Graphs", styles["Heading2"]))
    for key, caption in [
        ("profile","Figure 1. Raw and time-zero corrected pressure versus TVD."),
        ("pressure_time","Figure 2. Downhole pressure versus elapsed time, grouped by station."),
        ("rates","Figure 3. Station-wise regression buildup rates."),
        ("gradients","Figure 4. Raw and corrected pressure gradients versus TVD.")
    ]:
        story.append(Paragraph(caption, styles["Small"]))
        story.append(RLImage(fig_png(figs[key]), width=235*mm, height=132*mm))
        story.append(Spacer(1, 3*mm))

    story.append(Paragraph("Wellhead Shut-in Pressure (WHSIP) Soft QC", styles["Heading2"]))
    if meta["whsip_psi"] is not None:
        qc_text = (
            f"Input WHSIP at time zero: {meta['whsip_psi']:.3f} psi. "
            f"Linear extrapolation of corrected pressure-depth data to TVD=0: "
            f"{meta['extrapolated_whsip_psi']:.3f} psi. Difference (extrapolated minus input): "
            f"{meta['whsip_difference_psi']:.3f} psi. This is an indicative soft QC only; a single straight-line "
            "extrapolation may not represent a multi-fluid or piecewise-gradient wellbore."
        )
    else:
        qc_text = "No WHSIP value was supplied. WHSIP soft QC was not performed."
    story.append(Paragraph(qc_text, styles["BodyText"]))
    story.append(Spacer(1, 3*mm))

    story.append(Paragraph("Warnings and Data Checks", styles["Heading2"]))
    if meta["warnings"]:
        for w in meta["warnings"]:
            story.append(Paragraph("• " + w.replace("&","&amp;"), styles["Small"]))
    else:
        story.append(Paragraph("No automatic warning was triggered. This does not guarantee that the input data or engineering assumptions are correct.", styles["Small"]))
    if notes.strip():
        story.append(Spacer(1, 2*mm))
        story.append(Paragraph("Engineer Notes", styles["Heading2"]))
        story.append(Paragraph(notes.replace("&","&amp;").replace("\n","<br/>"), styles["BodyText"]))

    story.append(Spacer(1, 5*mm))
    story.append(Paragraph("Engineering assumptions and limitations", styles["Heading2"]))
    limitations = (
        "This report is based on the uploaded dataset and the stated station boundaries/timing. The method assumes that "
        "pressure buildup over each short station window can be approximated as linear. Correction durations must represent "
        "the intended elapsed interval, including movement and recording time, without gaps or double counting. The software "
        "uses the paper's cumulative station-rate approach; it does not prove that the reconstructed values equal a fully "
        "stabilized survey. Strong crossflow, wellbore storage, nonlinear transients, gauge drift, poor station boundaries, "
        "or insufficient readings can reduce reliability. Temperature is displayed when available but is not included in the "
        "core pressure-correction equation described in the supplied paper. Independent stabilized reference data are needed "
        "to quantify actual correction accuracy."
    )
    story.append(Paragraph(limitations, styles["Small"]))
    doc.build(story)
    buff.seek(0)
    return buff

st.title("📈 Static Gradient Survey (SGS) Analysis Software")
st.caption("Transient pressure correction using station-wise linear regression and cumulative time-zero referencing")
with st.sidebar:
    st.header("Analysis Settings")
    correction_factor = st.number_input("Global correction factor", min_value=0.0, value=1.0, step=0.05, format="%.3f",
        help="Multiplies every station buildup rate. Default 1.0 means no global adjustment.")
    project_name = st.text_input("Project / Well name", value="SGS Transient Pressure Correction")
    whsip_override_text = st.text_input("WHSIP at time zero (psi, optional)", value="")
    notes = st.text_area("Engineer notes (optional)", height=100)

st.markdown("""
Upload an SGS pressure-time dataset. The application calculates station buildup rates, cumulative transient corrections,
time-zero corrected pressures, pressure gradients, diagnostic plots and a downloadable PDF report.
""")
uploaded = st.file_uploader("Upload SGS input file", type=["csv", "xlsx", "xls"])

if uploaded is None:
    st.info("Upload your CSV/Excel file to begin. A sample 14-station dataset is included with the project package.")
    st.subheader("Required input columns")
    st.code("Station_ID, Time_min, Pressure_psi, TVD_ft")
    st.write("Recommended additional columns: Station_Start_Time_min, Station_End_Time_min, Station_Measurement_Time_min, Correction_Duration_min, WHSIP_psi_at_TimeZero, Temperature_C.")
    st.stop()

try:
    raw = read_input(uploaded)
    st.subheader("Uploaded data preview")
    st.write(f"{len(raw)} rows · {raw['Station_ID'].nunique() if 'Station_ID' in raw else 0} station IDs")
    st.dataframe(raw.head(20), use_container_width=True)

    missing = [c for c in REQUIRED if c not in raw.columns]
    if missing:
        st.error("Missing required columns: " + ", ".join(missing))
        st.stop()

    whsip_override = None
    if whsip_override_text.strip():
        whsip_override = float(whsip_override_text)

    data, result, meta = analyze_sgs(raw, correction_factor, whsip_override)
    figs = make_figures(data, result)

    st.success("SGS analysis completed.")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Stations analyzed", meta["station_count"])
    c2.metric("Raw data rows", meta["raw_rows"])
    c3.metric("Time-zero", f'{meta["time_zero_min"]:.2f} min')
    c4.metric("Correction factor", f'{meta["correction_factor"]:.3f}')

    if meta["warnings"]:
        for w in meta["warnings"]:
            st.warning(w)

    st.subheader("Station-wise calculation results")
    st.dataframe(result, use_container_width=True)

    st.subheader("Diagnostic graphs")
    st.pyplot(figs["profile"], use_container_width=True)
    st.pyplot(figs["pressure_time"], use_container_width=True)
    st.pyplot(figs["rates"], use_container_width=True)
    st.pyplot(figs["gradients"], use_container_width=True)

    st.subheader("WHSIP soft quality control")
    if meta["whsip_psi"] is not None:
        q1,q2,q3 = st.columns(3)
        q1.metric("Input WHSIP", f'{meta["whsip_psi"]:.2f} psi')
        q2.metric("Extrapolated pressure at TVD=0", f'{meta["extrapolated_whsip_psi"]:.2f} psi')
        q3.metric("Difference", f'{meta["whsip_difference_psi"]:.2f} psi')
        st.caption("Indicative soft QC only. A single linear extrapolation may not represent a multi-fluid or piecewise-gradient wellbore.")
    else:
        st.info("WHSIP not provided. Enter it in the sidebar or include WHSIP_psi_at_TimeZero in the input file.")

    csv_bytes = result.to_csv(index=False).encode("utf-8")
    pdf_bytes = build_pdf(result, meta, figs, project_name, notes).getvalue()
    d1,d2 = st.columns(2)
    d1.download_button("⬇ Download corrected results CSV", data=csv_bytes, file_name="SGS_Corrected_Results.csv", mime="text/csv", use_container_width=True)
    d2.download_button("⬇ Download complete PDF report", data=pdf_bytes, file_name="SGS_Analysis_Report.pdf", mime="application/pdf", use_container_width=True)

    st.caption("Research / engineering decision-support prototype. Review input quality, station timing, regression fit and assumptions before operational use.")

except Exception as e:
    st.error(f"Analysis could not be completed: {e}")
    st.info("Check the required columns, numeric values, station timestamps and at least 3 readings per station.")
