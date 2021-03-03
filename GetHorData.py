# -*- coding: utf-8 -*-

import requests
from bs4 import BeautifulSoup
import csv


yesterday = '20200302'
today = '20210302'

def listToString(s):  
    
    # initialize an empty string 
    str1 = ""  
    
    # traverse in the string   
    for ele in s:  
        str1 += ele   
    
    # return string   
    return str1


def GetHoroscopeData():
    
    horoscope = []
    
    signs = [] # each number represent one sign 
    for i in range(1,13,1):
        signs.append(i)
        
        
        
    # days per month [month,days]
    months = [['01',31],['02',28],['03',31],['04',30],['05',31],['06',30],['07',31],['08',31],['09',30],['10',31],['11',30],['12',31]]
        
    date = []
    
    for year in range(2020,2022,1):
        for month in months[:]:
            for days in range(1,month[1]+1,1):
                if days <10:
                    date.append(listToString([str(year),month[0],str(0),str(days)]))
                else:
                    date.append(listToString([str(year),month[0],str(days)]))
                    
    #print(date[date.index(yesterday):date.index(today)],len(date[date.index(yesterday):date.index(today)]))
    
    for sign in signs:
        for elem in date[date.index(yesterday):date.index(today)]:
            page = requests.get("https://www.horoscope.com/us/horoscopes/general/horoscope-archive.aspx?sign="+str(sign)+"&laDate="+elem)
            soup = BeautifulSoup(page.content, 'html.parser')
            s = soup.find_all('p')[0].get_text()
            horoscope.append([sign,elem,s])
            
    return horoscope
                  
with open('HoroscopeData.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(GetHoroscopeData())
    



