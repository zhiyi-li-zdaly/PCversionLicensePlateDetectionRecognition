# File operations to validate/filter license plate information.
import sys
import os
import time
import datetime
from datetime import datetime
import numpy as np
from difflib import SequenceMatcher
import shutil
import pandas as pd
import re
      
def has_numbers(inputString):
    return any(char.isdigit() for char in inputString)

def countCommonIn2Strings(string1, string2):
    n1 = len(string1)
    n2 = len(string2)
    p1 = 0
    p2 = 0
    res = 0
    while p1 < min(n1, n2):  
        if string1[p1] == string2[p2]:
            res += 1
        p1 += 1
        p2 += 1
    
    return res

def postProcessing():
    cur_time = datetime.now()
    midnight = datetime.combine(cur_time, datetime.max.time())
    # print("cur_time: ", cur_time)
    # print("midnight: ", midnight)
    cur_date = str(cur_time.date())
        
    # postprocess happens here. 
    #  Copy log file to another file and open it
    log_origin_name = './samples/log.txt'
    log_temp_name = './samples/log_temp.txt'

    if os.path.exists(log_temp_name):
        os.remove(log_temp_name)
                 
    shutil.copy2(log_origin_name, log_temp_name)
    print("copy file successful")

    # Create a dictionary to store related information
    licensePlate_Dic = {}	# Key: First license plate read so far. 
                        # Value: Dic {} with key2: license plate read in current, value2: Whole current line read.  
    # Open the temporary file
    file1 = open(log_temp_name, 'r')
    line = file1.readline()

    outFile1 = open("./samples/test1.txt","w")
    for line in file1.readlines():
        items = line.rstrip("\n").split(",")
   
        # Get timestamp
        timestamp_Str = items[1]
        # print("timestamp_Str: ", timestamp_Str)
        timestamp_Str_date = timestamp_Str.split(" ")[0]
        print("timestamp_Str_date: ", timestamp_Str_date)
        print("cur_date: ", cur_date)

        # if timestamp_Str_date != '2024-04-14':
        if timestamp_Str_date != cur_date: # Only record current date
            continue  

        # Filter and validate licensePlate.
        # Skip license plate is missing
        if len(items) < 3:
            continue
    
        cur_licensePlate = items[2].strip()
        # Skip licensePlate which is empty
        if cur_licensePlate == "": # S
            continue
    
        if len(cur_licensePlate) <= 3:
            continue

        #if not cur_licensePlate[-1].isalnum():
        #    continue

        # if not cur_licensePlate[0].isalnum():
        #    continue
    
        last4Digits = cur_licensePlate[-4:]
        # if not last4Digits.isdigit():
        #    continue

        non4Digits = cur_licensePlate[:-4]
        # if has_numbers(non4Digits):
        #     continue

        # Skip licensePlate has empty space
        # res = bool(re.search(r"\s", last4Digits))
        res = bool(re.search(r"\s", cur_licensePlate))
        # if res == True:
        #     continue

        # print("cur_licensePlate: ", cur_licensePlate)
        # Remove non-alpha

        if cur_licensePlate in licensePlate_Dic: # Exact match just save 
            licensePlate_Dic[cur_licensePlate][cur_licensePlate].append(line)
            print("Exact match found")
            # print(licensePlate_Dic)
            
        else:	# Not exact match 
            # print("Not exact match found")
            if not licensePlate_Dic:	# Dictionary is empty 
                # print("Dictionary is empty")
                licensePlate_Dic[cur_licensePlate] = {}
                licensePlate_Dic[cur_licensePlate][cur_licensePlate] = []
                licensePlate_Dic[cur_licensePlate][cur_licensePlate].append(line)
                # print(licensePlate_Dic)
                # sys.exit(0)

            else:	# Dictionary not empty
                # Through dictionary to search for similar one
                # print("Dictionary not empty")
                found = False
                found_key = ""
                for key1 in licensePlate_Dic.keys():
                    # print(cur_licensePlate, key1)
                    # sys.exit(0)
                
                    # Ignore separation of licensePlate and compare with each other
                    candidate_cur_licensePlate = cur_licensePlate[:3] + cur_licensePlate[4:]
                    candidate_key1 = key1[:3] + key1[4:]
                    # match = SequenceMatcher(None, cur_licensePlate, key1).find_longest_match()
                    match = SequenceMatcher(None, candidate_cur_licensePlate, candidate_key1).find_longest_match()
                    sumInCommonSize = countCommonIn2Strings(cur_licensePlate, key1)

                    # print("match.size: ",  match.size)
                    # print("sumInCommonSize: ",  sumInCommonSize)

                    if match.size >= 3 or sumInCommonSize >= 4:
                        # print("match.size: ", match.size)
                        # sys.exit(0)
                        # print("cur_licensePlate: ", cur_licensePlate)
                        # print("key1: ", key1)
                        found = True
                        found_key = key1
                        break
           
                # print("found: ", found)
                # print("cur_licensePlate: ", cur_licensePlate)
                # sys.exit(0)
            
                if found == True:
                    found_level2 = False
                    for key2 in licensePlate_Dic[found_key].keys():
                        if cur_licensePlate == key2:
                            licensePlate_Dic[found_key][key2].append(line)
                            found_level2 = True
                            break
                    
                    if found_level2 == False:
                        licensePlate_Dic[found_key][cur_licensePlate] = []
                        licensePlate_Dic[found_key][cur_licensePlate].append(line)
         
                    # print(licensePlate_Dic)
                    # sys.exit(0)

                else: # Not found in the dictionary, establish new records
                        licensePlate_Dic[cur_licensePlate] = {}
                        licensePlate_Dic[cur_licensePlate][cur_licensePlate] = []
                        licensePlate_Dic[cur_licensePlate][cur_licensePlate].append(line)
           
        outFile1.write(line)

    # print(licensePlate_Dic.keys())
    # print(licensePlate_Dic)
                      
    record_file_name = "./samples/timeRecordLicensePlateInformation_v5_test1.txt"
    check_file = os.path.exists(record_file_name)  
    if not check_file: 
        with open(record_file_name, "w") as f:
            outLine = "Station_id" + "," + "timestamp" + "," + "licenseplate" + "\n"
            f.write(outLine)
            f.close()
                   
    outputFile2 = open(record_file_name, "a")
    # print("licensePlate_Dic: ", licensePlate_Dic)

    
    for key in licensePlate_Dic.keys():
        temp_Dic = licensePlate_Dic[key]
        max_key = ""
        max_val = 0
        for key2 in temp_Dic.keys():
            value2 = temp_Dic[key2]
            if len(value2) >= max_val:
                max_val = len(value2)
                max_key = key2
    
        outLine = licensePlate_Dic[key][max_key][0]
        outputFile2.write(outLine)

    outputFile2.close() 
    file1.close()
    outFile1.close()
    return





