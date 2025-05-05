import csv
import utm
import matplotlib.pyplot as plt

############### Exercise 9.1.1 ###############

# --- Config ---
log_file = "mini_project_2/Data for miniproject on visual odometry/DJIFlightRecord_2021-03-18_[13-04-51]-TxtLogToCsv.csv"
simulate_25fps = True  # Set to True to simulate 25fps (sample every 25th + first)

# --- Data Storage ---
latitudes = []
longitudes = []
utm_x = []
utm_y = []

# --- Read CSV ---
with open(log_file, 'r', encoding='utf-8', errors='replace') as f:
    reader = csv.DictReader(f)
    for row in reader:
        try:
            lat = float(row['OSD.latitude'])
            lon = float(row['OSD.longitude'])
            if lat != 0 and lon != 0:
                latitudes.append(lat)
                longitudes.append(lon)
        except (ValueError, KeyError):
            continue

# --- Downsample for 25fps simulation ---
if simulate_25fps:
    latitudes = latitudes[::25]
    longitudes = longitudes[::25]

# --- Convert to UTM ---
for lat, lon in zip(latitudes, longitudes):
    x, y, _, _ = utm.from_latlon(lat, lon)
    utm_x.append(x)
    utm_y.append(y)

# --- Plot ---
plt.figure(figsize=(10, 6))
plt.plot(utm_x, utm_y, color='blue')
plt.title('UAV Flight Path (UTM)')
plt.xlabel('UTM X')
plt.ylabel('UTM Y')
plt.grid(True)
plt.axis('equal')
plt.savefig("mini_project_2/output/flightpath_utm.png")