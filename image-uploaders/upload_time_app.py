
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Upload Time Calculator", layout="centered")

st.title("Cloud Upload Time Calculator")

# Sidebar controls
st.sidebar.header("Inputs")
file_size_gb_str = st.sidebar.text_input("File size (GB)", value="7.8")
min_speed = st.sidebar.number_input("Min upload speed (Mbps)", min_value=1, max_value=10000, value=10, step=1)
max_speed = st.sidebar.number_input("Max upload speed (Mbps)", min_value=min_speed, max_value=10000, value=120, step=1)
step_speed = st.sidebar.number_input("Speed increment (Mbps)", min_value=1, max_value=10000, value=10, step=1)

st.sidebar.markdown("---")
st.sidebar.subheader("Throttle levels")
throttle_70 = st.sidebar.checkbox("70% of link", value=True)
throttle_80 = st.sidebar.checkbox("80% of link", value=True)
throttle_90 = st.sidebar.checkbox("90% of link", value=True)
throttle_100 = st.sidebar.checkbox("100% of link", value=True)

selected_throttles = []
if throttle_70: selected_throttles.append(0.7)
if throttle_80: selected_throttles.append(0.8)
if throttle_90: selected_throttles.append(0.9)
if throttle_100: selected_throttles.append(1.0)

st.sidebar.markdown("---")
unit_note = st.sidebar.checkbox("Use decimal GB (1 GB = 1000 MB)", value=True,
    help="When checked, 1 GB = 1000 MB = 8000 megabits. Uncheck to use GiB (1 GiB = 1024 MiB).")

# Validate inputs
try:
    file_size_gb = float(file_size_gb_str)
    if file_size_gb <= 0:
        st.error("File size must be greater than 0.")
        st.stop()
except ValueError:
    st.error("File size must be a number.")
    st.stop()

if max_speed < min_speed or step_speed <= 0:
    st.error("Please ensure max speed ≥ min speed and increment > 0.")
    st.stop()

if not selected_throttles:
    st.error("Select at least one throttle level.")
    st.stop()

# Convert to megabits
if unit_note:
    # Decimal GB: 1 GB = 1000 MB -> 8000 megabits
    file_size_megabits = file_size_gb * 8000.0
else:
    # GiB: 1 GiB = 1024 MiB -> 8192 megabits
    file_size_megabits = file_size_gb * 8192.0

speeds_mbps = np.arange(min_speed, max_speed + 1, step_speed)

# Compute times
data = {"Upload Speed (Mbps)": speeds_mbps}
for t in selected_throttles:
    # time (minutes) = size (megabits) / (Mbps * throttle) / 60
    times_minutes = (file_size_megabits / (speeds_mbps * t)) / 60.0
    data[f"{int(t*100)}% of link"] = times_minutes

df = pd.DataFrame(data)

# Title and context
st.markdown(
    f"""
**File size:** {file_size_gb:g} {'GB' if unit_note else 'GiB'}  
**Speeds (Mbps):** {min_speed} to {max_speed} in steps of {step_speed}  
**Throttle levels:** {", ".join([f"{int(t*100)}%" for t in selected_throttles])}
"""
)

# Plot
fig, ax = plt.subplots(figsize=(9, 5.5))
for t in selected_throttles:
    label = f"{int(t*100)}% of link"
    ax.plot(df["Upload Speed (Mbps)"], df[label], marker="o", label=label)

ax.set_title(f"Time to Upload {file_size_gb:g} {'GB' if unit_note else 'GiB'} vs Upload Speed")
ax.set_xlabel("Upload Speed (Mbps)")
ax.set_ylabel("Time to Upload (minutes)")
ax.grid(True, linestyle="--", alpha=0.7)
ax.legend()
st.pyplot(fig, clear_figure=True)

# Show table
st.subheader("Calculated Times (minutes)")
st.dataframe(df, use_container_width=True)

# Download CSV
csv_bytes = df.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download CSV",
    data=csv_bytes,
    file_name=f"upload_times_{file_size_gb:g}{'GB' if unit_note else 'GiB'}.csv",
    mime="text/csv",
)
