from flask import Flask, render_template
import pandas as pd
from datetime import datetime

app = Flask(__name__)

@app.route("/")
def index():
    try:
        df = pd.read_csv("attendance.csv")
        df = df.dropna(how="all")
        data = df.to_dict(orient="records")
        total_logs = len(df)
        unique_doctors = int(df["Name"].nunique()) if "Name" in df.columns else 0
        last_entry = df["Entry"].dropna().iloc[-1] if "Entry" in df.columns and not df["Entry"].dropna().empty else "-"
        last_exit = df["Exit"].dropna().iloc[-1] if "Exit" in df.columns and not df["Exit"].dropna().empty else "-"
        last_updated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        data = []
        total_logs = 0
        unique_doctors = 0
        last_entry = "-"
        last_exit = "-"
        last_updated = "-"

    return render_template(
        "index.html",
        data=data,
        total_logs=total_logs,
        unique_doctors=unique_doctors,
        last_entry=last_entry,
        last_exit=last_exit,
        last_updated=last_updated,
    )

if __name__ == "__main__":
    app.run(debug=True)
