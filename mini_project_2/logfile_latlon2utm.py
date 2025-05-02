import csv
import utm
import matplotlib.pyplot as plt

# --- Config ---
log_file = "Data for miniproject on visual odometry/DJIFlightRecord_2021-03-18_[13-04-51]-TxtLogToCsv.csv"  # Replace with your log filename

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
            if lat != 0 and lon != 0:  # filter out invalid GPS data
                latitudes.append(lat)
                longitudes.append(lon)
                x, y, _, _ = utm.from_latlon(lat, lon)
                utm_x.append(x)
                utm_y.append(y)
        except (ValueError, KeyError):
            continue  # skip rows with bad/missing data

# --- Plot ---
plt.figure(figsize=(10, 6))
plt.plot(utm_x, utm_y, color='blue')
plt.title('UAV Flight Path (UTM)')
plt.xlabel('UTM X')
plt.ylabel('UTM Y')
plt.grid(True)
plt.axis('equal')
plt.savefig("output/flightpath_utm.png")