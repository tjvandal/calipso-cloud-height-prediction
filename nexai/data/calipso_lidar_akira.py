"""CALIPSO Lidar"""

# Author: Akira Sewnath

from pyhdf.SD import SD, SDC
import numpy as np
import os

class CalipsoLidar():
    
    #LAT LON BOUNDARY ADDITION
    
    def __init__(self, calipsoFile):
        self.calipsoFile = calipsoFile
        self.orgTimes = []
        self.convertedTimes = []
        self.xABI = []
        self.yABI = []
        self.cloudTopHeight = []
        self.orgCTH = []
        
        #new addition for noaal1b
        self.dayOfYear = 0
        self.convertedTime = 0
        self.hours = []
        self.year = 0
        self.minutes = []
        self.seconds = []
        self.years = []
        self.daysOfYear = []
        
    def toABISpace(self, projDict, goesMinMax):
        calDataset = SD(self.calipsoFile, SDC.READ)
        degLat = calDataset.select('Latitude')
        degLat = degLat[:]
        degLon = calDataset.select('Longitude')
        degLon = degLon[:]
        
        #radiances   
        cth = calDataset.select('Layer_Top_Altitude')
        cth = cth[:]
                
        r_pol = projDict["r_pol"]
        r_eq  = projDict["r_eq"]
        e     = projDict["e"]
        H     = projDict["H"]
        r_frac_inv  = projDict["r_frac_inv"]
        lambda_zero = projDict["lambda_zero"]
        
        xGoesMax = goesMinMax["xGoesMax"]
        xGoesMin = goesMinMax["xGoesMin"]
        yGoesMax = goesMinMax["yGoesMax"]
        yGoesMin = goesMinMax["yGoesMin"]
        
        time = calDataset.select('Profile_UTC_Time')[:]
        for ind in range(len(degLat)):
            
            if( (degLat[ind] >= -50 ) and (degLat[ind] <= 50) and 
              (degLon[ind] >= -105 ) and (degLon[ind] <= -35) ):
        
                lat = degLat[ind][0] * (np.pi/180)
                lon = degLon[ind][0] * (np.pi/180)
            
                centerLat = np.arctan(r_frac_inv * np.tan(lat))
                r_c = r_pol / np.sqrt(1 - (e**2)*(np.cos(centerLat)**2))
                s_x = H - ( r_c * np.cos(centerLat) * np.cos(lon-lambda_zero) )
                s_y = -r_c*np.cos(centerLat)*np.sin(lon-lambda_zero)
                s_z = r_c*np.sin(centerLat)
                r_frac = (r_eq**2)/(r_pol**2)
                inequality_left_term = H * (H-s_x)
                inequality_right_term  = (s_y**2) + ( r_frac * (s_z**2) )
            
                if( ~(inequality_left_term < inequality_right_term) ):
                    y = np.arctan( (s_z/s_x) )
                    x = np.arcsin( -s_y / np.sqrt( (s_x**2) + (s_y**2) + (s_z**2) ) )
                    
                    #do another filter here to assess whether it is in the GOES16 bounds        
                    if( (xGoesMin <= x <= xGoesMax) and (yGoesMin <= y <= yGoesMax) ):
            
                        #These are the converted CALIPSO coordinates that potentially match to a GOES point
                        self.xABI.append(x)
                        self.yABI.append(y)
                        self.orgTimes.append(time[ind])
                        self.orgCTH.append(cth[ind][0])
                        
    def notEmpty(self):
        if(not self.xABI or not self.yABI or not self.orgTimes):
            return False
        else:
            return True
        
        
    def clear(self): 
        self.cloudTopHeight = []
        self.xABI = []
        self.yABI = []
        self.orgTimes = []
        self.orgCTH = []
        self.convertedTimes = []
        
        
    def findDayOfYear(self, date):
        #Calculation to find day of year. Assuming day doesn't change
        
        #Account for leap year eventually
        monthDays = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]  
        
        #extract month and day
        month = int(date[0:2])
        day   = int(date[2:4])
        
        #day summation
        return np.sum(monthDays[0:month-1]) + day
    
      
    def setMasterArguments(self):
        #Get year
        filename = os.path.basename(self.calipsoFile)
        print(self.calipsoFile)
        datetime = filename.split('.')[1]
        self.year = int(datetime[0:4])
        
        #Find day of year (pass date as a string) extract directly from filename
        #date = str(self.orgTimes[0])[3:7]
        date = datetime[5:7] + datetime[8:10]
        self.dayOfYear = self.findDayOfYear(date)
        
        #Set master hour extract directly from filename
        self.masterHour = int(datetime[11:13])
    
        
    def temporalTimeConversion(self):
        #Convert all CALIPSO times to be compared to GOES (for each CALIPSO file)
        
        #Create time strings for spatial collocation
        #must also set DayOfYear and Year
        for time in self.orgTimes:
            
            time_str = str(time[0])
            d_year = int("20" + time_str[0:2])
            d_dayOfYear = self.findDayOfYear(time_str[2:6])
            
            d_dec = time - np.fix(time)
            hour_term = d_dec*24 
            d_hour = int(hour_term) 
            d_min = int( (hour_term - np.fix(hour_term)) * 60)
            
            self.minutes.append(d_min)
            self.hours.append(d_hour)
            self.years.append(d_year)
            self.daysOfYear.append(d_dayOfYear)  
        
    def getConvertedTimes(self):
        return self.convertedTimes
    
    def getXArr(self):
        return self.xABI
    
    def getYArr(self):
        return self.yABI
    
    def getCloudTopHeight(self):
        calDataset = SD(self.calipsoFile, SDC.READ)
        cloudTopHeight = calDataset.select('Layer_Top_Altitude')
        cloudTopHeight = cloudTopHeight[:]
        return cloudTopHeight
