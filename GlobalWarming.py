#import pylab
import numpy as np
import re
import matplotlib.pyplot as plt
# cities in our weather data
CITIES = [
    'BOSTON',
    'SEATTLE',
    'SAN DIEGO',
    'PHILADELPHIA',
    'PHOENIX',
    'LAS VEGAS',
    'CHARLOTTE',
    'DALLAS',
    'BALTIMORE',
    'SAN JUAN',
    'LOS ANGELES',
    'MIAMI',
    'NEW ORLEANS',
    'ALBUQUERQUE',
    'PORTLAND',
    'SAN FRANCISCO',
    'TAMPA',
    'NEW YORK',
    'DETROIT',
    'ST LOUIS',
    'CHICAGO'
]

INTERVAL_1 = list(range(1961, 2006))
INTERVAL_2 = list(range(2006, 2016))
x3=INTERVAL_1+INTERVAL_2

"""
Begin helper code
"""
class Climate(object):
    """
    The collection of temperature records loaded from given csv file
    """
    def __init__(self, filename):
        """
        Initialize a Climate instance, which stores the temperature records
        loaded from a given csv file specified by filename.

        Args:
            filename: name of the csv file (str)
        """
        self.rawdata = {}

        f = open(filename, 'r')
        header = f.readline().strip().split(',')
        for line in f:
            items = line.strip().split(',')

            date = re.match('(\d\d\d\d)(\d\d)(\d\d)', items[header.index('DATE')])
            year = int(date.group(1))
            month = int(date.group(2))
            day = int(date.group(3))

            city = items[header.index('CITY')]
            temperature = float(items[header.index('TEMP')])
            if city not in self.rawdata:
                self.rawdata[city] = {}
            if year not in self.rawdata[city]:
                self.rawdata[city][year] = {}
            if month not in self.rawdata[city][year]:
                self.rawdata[city][year][month] = {}
            self.rawdata[city][year][month][day] = temperature
            
        f.close()

    def get_yearly_temp(self, city, year):
        """
        Get the daily temperatures for the given year and city.

        Args:
            city: city name (str)
            year: the year to get the data for (int)

        Returns:
            a numpy 1-d array of daily temperatures for the specified year and
            city
        """
        temperatures = []
        assert city in self.rawdata, "provided city is not available"
        assert year in self.rawdata[city], "provided year is not available"
        for month in range(1, 13):
            for day in range(1, 32):
                if day in self.rawdata[city][year][month]:
                    temperatures.append(self.rawdata[city][year][month][day])
        return np.array(temperatures)

    def get_daily_temp(self, city, month, day, year):
        """
        Get the daily temperature for the given city and time (year + date).

        Args:
            city: city name (str)
            month: the month to get the data for (int, where January = 1,
                December = 12)
            day: the day to get the data for (int, where 1st day of month = 1)
            year: the year to get the data for (int)

        Returns:
            a float of the daily temperature for the specified time (year +
            date) and city
        """
        assert city in self.rawdata, "provided city is not available"
        assert year in self.rawdata[city], "provided year is not available"
        assert month in self.rawdata[city][year], "provided month is not available"
        assert day in self.rawdata[city][year][month], "provided day is not available"
        return self.rawdata[city][year][month][day]



"""
End helper code
"""

# Problem 1
def generate_models(x, y, degs):
    """
    Generate regression models by fitting a polynomial for each degree in degs
    to points (x, y).
    Args:
        x: a list with length N, representing the x-coords of N sample points
        y: a list with length N, representing the y-coords of N sample points
        degs: a list of degrees of the fitting polynomial
    Returns:
        a list of numpy arrays, where each array is a 1-d array of coefficients
        that minimizes the squared error of the fitting polynomial
    """
    output=[]
    for degree in degs:
        output.append(np.polyfit(x,y,degree))
    return output

#a=[5,6,7,8]
#b=[115,107,129,155]
#generate_models(a, b, [1])  --> [array([ 14.2,  34.2])]
#f(x) = 14.2*x+34.2
#c=[]
#for x in a: c.append(f(x))
#c=[105.2, 119.39999999999999, 133.6, 147.8]



    
    
# Problem 2
def r_squared(y, estimated):
    """
    Calculate the R-squared error term.
    Args:
        y: list with length N, representing the y-coords of N sample points
        estimated: a list of values estimated by the regression model
    Returns:
        a float for the R-squared error term
    """
    
    avey=sum(y)/len(y)
    numerator=0
    denominator=0
    i=0
    while i<len(y):
        numerator+=((y[i]-estimated[i])**2)
        denominator+=((y[i]-avey)**2)
        i+=1
    return 1-(numerator/denominator)

#a=[5,6,7,8]
#b=[115,107,129,155]
#generate_models(a, b, [1])  --> [array([ 14.2,  34.2])]
#f(x) = 14.2*x+34.2
#c=[]
#for x in a: c.append(f(x))
#c=[105.2, 119.39999999999999, 133.6, 147.8]
#print(r_squared(b,c)) --> 0.757475582268971



#THIS IS A HELPER FUNCTION
def get_ests_from_model(x,model):
    """
    x is list
    model is numpy.1darray
    returns a list ests of the estimated outputs based on this model
    """
    f=np.poly1d(model)
    ests=[]
    for v in x:
        ests.append(f(v))
    return (ests)

#a=[5,6,7,8]
#b=[115,107,129,155]
#generate_models(a, b, [1])  --> [array([ 14.2,  34.2])]
#f(x) = 14.2*x+34.2
#c=[]
#for x in a: c.append(f(x))
#c=[105.2, 119.39999999999999, 133.6, 147.8]
#get_ests_from_model(a,c[0]) --> [105.19999999999993, 119.39999999999995,
#133.59999999999997, 147.79999999999998]




    
# Problem 3
def evaluate_models_on_training(x, y, models):
    """
    For each regression model, compute the R-square for this model with the
    standard error over slope of a linear regression line (only if the model is
    linear), and plot the data along with the best fit curve.

    For the plots, you should plot data points (x,y) as blue dots and your best
    fit curve (aka model) as a red solid line. You should also label the axes
    of this figure appropriately and have a title reporting the following
    information:
        degree of your regression model,
        R-square of your model evaluated on the given data points
    Args:
        x: a list of length N, representing the x-coords of N sample points
        y: a list of length N, representing the y-coords of N sample points
        models: a list containing the regression models you want to apply to
            your data. Each model is a numpy array storing the coefficients of
            a polynomial.
    Returns:
        None
    """
    for model in models:
        ests=get_ests_from_model(x,model)
        r_sq=r_squared(y,ests)
        #print(r_sq)
        plt.scatter(x,y,color='blue')
        plt.xlim(x[0],x[len(x)-1])
        plt.plot(x,ests,color="red")
        plt.title("Data Blue and FitLine Red for degree "+str(len(model)-1)+"\n"+
        "and R-Squared value "+str(r_sq))
        plt.xlabel("Year")
        plt.ylabel("Temperature in Celsius")
        plt.show()

        
        
        
### Begining of program
raw_data = Climate('data.csv')

"""
# Problem 3
y = []
x = INTERVAL_1
for year in INTERVAL_1:
    y.append(raw_data.get_daily_temp('BOSTON', 1, 10, year))   
models = generate_models(x, y, [1])
#models = generate_models(x, y, [1,2,3,4])
#print(models)
evaluate_models_on_training(x, y, models)



# Problem 4: FILL IN MISSING CODE TO GENERATE y VALUES
x1 = INTERVAL_1
x2 = INTERVAL_2
y = []
for year in x1:
    y.append(np.mean(raw_data.get_yearly_temp("BOSTON",year)))
models = generate_models(x1, y, [1])    
evaluate_models_on_training(x1, y, models)
"""



#EXPAND DATA TO NOW APPLY TO ALL 21 CITIES OVER THE ENTIRE RANGE (1961-2015)
#*******************MEGA OVERALL GLOBAL WARMING ANALYSIS***************

y=[]   
for year in x3:
    avesholder=[]
    for city in CITIES:   
        avesholder.append(np.mean(raw_data.get_yearly_temp(city,year)))
    #print(str(len(avesholder))+" this should be 21 each time...")
    aveOverallForYear=sum(avesholder)/len(CITIES)
    #print (aveOverallForYear)
    y.append(aveOverallForYear)
#print(y)
    
models=generate_models(x3,y,[1])
evaluate_models_on_training(x3,y,models)
        

