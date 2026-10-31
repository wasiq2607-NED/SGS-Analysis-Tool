# SGS Analysis Software

A Streamlit application for Static Gradient Survey (SGS) transient pressure correction.

## Features
- CSV and Excel upload
- Station-wise least-squares linear regression
- Station buildup rate and R-squared
- Cumulative pressure correction
- Time-zero corrected station pressure
- Raw and corrected pressure-depth plots
- Pressure-time plot and buildup-rate plot
- Raw and corrected pressure gradients
- Optional WHSIP soft QC
- Downloadable corrected-results CSV
- Complete PDF report with calculation table, graphs, QC and limitations

## Required input columns
- `Station_ID`
- `Time_min`
- `Pressure_psi`
- `TVD_ft`

## Recommended additional columns
- `Station_Start_Time_min`
- `Station_End_Time_min`
- `Station_Measurement_Time_min`
- `Correction_Duration_min`
- `WHSIP_psi_at_TimeZero`
- `Temperature_C`

The included sample CSV contains all recommended columns.

## Run locally
1. Install Python 3.10 or newer.
2. Open a terminal in this folder.
3. Install dependencies:

   `pip install -r requirements.txt`

4. Start the application:

   `streamlit run app.py`

5. Open the local URL printed by Streamlit (usually http://localhost:8501).

## Deploy for other users
Push this folder to a GitHub repository, then create a Streamlit Community Cloud app and select `app.py` as the entry point. Share the deployed app URL with other users.

## Method and important engineering notes
The application fits `P(t) = a + b*t` separately for each station using least squares. The slope `b` is the station buildup rate. The pressure contribution is `b * correction duration`, and contributions are cumulatively summed. Corrected pressure is measured pressure minus cumulative correction.

Time zero is the start of the first station's recording. The correction duration for the first station therefore runs from time zero to its measurement time; later durations run from the previous station measurement time to the current station measurement time. Supplied durations are used as provided. The app warns if their sum differs materially from elapsed time.

This implements the cumulative station-rate approach described in the supplied SGS paper. The method assumes short station buildup trends are approximately linear. Movement-time attribution must be justified by the field timing records and should not be double-counted.

WHSIP comparison is a soft QC. The app reports a linear extrapolation of corrected pressure-depth data to TVD=0, which is only indicative and may not represent a multi-fluid or piecewise-gradient wellbore.

Temperature is retained/displayed as an input field but is not used in the core pressure-correction equation in the supplied paper.

This prototype does not prove equivalence to a fully stabilized survey. Independent stabilized reference data are needed to quantify accuracy. Review station boundaries, gauge quality, regression fit, and all engineering assumptions before operational use.
