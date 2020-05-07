# Bird migration tracking using GPS

import pandas as pd

birddata = pd.read_csv('bird_tracking.csv', index_col = 0)
birddata.head()
birddata.info()
Counter(birddata.bird_name) #Count of data points across the 3 Gulls

#Plotting to understand flight path
import matplotlib.pyplot as plt
import numpy as np
eric_data = birddata.bird_name=="Eric"

x_eric, y_eric = birddata.longitude[eric_data], birddata.latitude[eric_data]

plt.plot(x_eric, y_eric, ".")

	#For all 3 birds
bird_names = pd.unique(birddata.bird_name)
plt.figure(figsize=(7,7))
for bird_name in bird_names:
	eric_data = birddata.bird_name==bird_name
	x_eric, y_eric = birddata.longitude[eric_data], birddata.latitude[eric_data]
	plt.plot(x_eric, y_eric, ".", label = bird_name)
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.legend(loc = "lower right")


#Investigating bird speed
eric_data = birddata.bird_name=="Eric"

speed = birddata.speed_2d[eric_data]
plt.hist(speed) #Warning stating that some values are not a number

speed_nan = np.isnan(speed)
sum(speed_nan) #85 entries for the Gull Eric alone are not a number


plt.figure(figsize = (8,4))
plt.hist(speed[~speed_nan], bins = np.linspace(0,30,30), density = True) #This is the same plotting command as writtien a few lines ago but the indices are filtered to include only numbers. It also says that 30 bins need to be used and the frequency is normalized to sum up to 1 instead of 10000, etc.
	#NOTE:  If x is a bool vector, ~x is the opposite or the NOT of x
plt.xlabel("2D Speed (m/s)")
plt.ylabel("Frequency")


#Checking out the datetime object
import datetime
datetime.datetime.today()
time_1 = datetime.datetime.today()
time_2 = datetime.datetime.today()
time_2-time_1


#Adding timestamp column to the birddata df and converting date_time present in the data to datetime format
timestamps = []

for k in range(len(birddata.date_time)):
    timestamps.append(datetime.datetime.strptime\
    (birddata.date_time.iloc[k][:-3], "%Y-%m-%d %H:%M:%S"))

birddata["timestamp"] = pd.Series(timestamps, index = birddata.index)

#Adding an elapsed time field

	#Creating a idctionary with min bird times for each of the 3 birds
min_times = []
for bird in bird_names:
	min_times.append(min(birddata.timestamp[birddata.bird_name == bird]))

bird_min_time_dict = dict(zip(bird_names, min_times)) #Dictionary with 3 birds and their min times

bird_min_time_list = []
for i in range(len(birddata.bird_name)):
	bird_min_time_list.append(bird_min_time_dict[birddata.bird_name[i]]) # list with same number of rows as birddata and each row containing min real time corresponding to each row's bird name

time_elapsed = []
for i in range(len(birddata)):
	time_elapsed.append(birddata.timestamp[i] - bird_min_time_list[i]) #Subtracting the time stamp and min real time to get time elapsed


#Computing daily mean speed
days_elapsed = np.array(time_elapsed)/datetime.timedelta(days=1)


next_day = 1
inds = []
daily_mean_speed = []

for (i,t) in enumerate(days_elapsed[birddata.bird_name == "Eric"]):
	if t < next_day:
		inds.append(i)
	else:
		next_day += 1
		daily_mean_speed.append(np.mean(birddata.speed_2d[inds]))
		inds = []

daily_mean_speeds_eric = np.array(daily_mean_speed)

#daily_mean_speeds_eric and other bird speeds have been done using the indices of the corresponding birds in the birddata file. This is because the above function stops working after reaching a new bird name

plt.figure(figsize=(10,10))
plt.plot(daily_mean_speeds_eric, "r", label = "eric")
plt.plot(daily_mean_speeds_nico, "b", label = "nico")
plt.plot(daily_mean_speeds_sanne, "y", label = "sanne")
plt.xlabel("days")
plt.ylabel("speed (m/s)")
plt.legend(loc = "upper right")




#Plotting the bird flights on a map
import cartopy.crs as ccrs
import cartopy.feature as cfeature

proj = ccrs.Mercator()
plt.figure(figsize=(12,12))

ax = plt.axes(projection = proj) # Modifies projeciton of x and y axes to fit the globe
ax.add_feature(cfeature.LAND) #Adds land and the other features to the map
ax.add_feature(cfeature.OCEAN)
ax.add_feature(cfeature.BORDERS)
ax.add_feature(cfeature.COASTLINE)
ax.set_extent((-25,20,52,10))

for bird_name in bird_names: #Following code is the same as the one above except with this one being on a map
	eric_data = birddata.bird_name==bird_name
	x_eric, y_eric = birddata.longitude[eric_data], birddata.latitude[eric_data]
	ax.plot(x_eric, y_eric, ".", transform = ccrs.Geodetic(), label = bird_name)

plt.legend(loc = "upper left")