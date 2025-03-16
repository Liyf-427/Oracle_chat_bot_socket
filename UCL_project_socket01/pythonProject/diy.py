from datetime import datetime, timedelta
from FuncA import pred_a
from FuncB import pred_b
from FuncC import pred_c

import os
import re
#global variable
# Part A weather
# Part B air quality in California
# extract data
# pre-define location
def get_file_path_from_address():
    """
    read from static folder/address.txt and return absolute path
    """
    # set file path
    file_path = "static/address.txt"

    # check existence
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ File {file_path} does not exist，check file path")

    # read
    with open(file_path, 'r') as file:
        address = file.readline().strip()  # extract file address and remove space
    return address

# self-checking
file_path = get_file_path_from_address()
print(f"Reading file from: {file_path}")

locations = ["Beijing", "Washington", "London", "Paris", "Berlin"] # Part A city List

# data format matching（yyyy/mm/dd）
date_pattern = r"(\d{4})[./\-](\d{1,2})[./\-](\d{1,2})"  # support yyyy.mm.dd or yyyy/mm/dd format

# the main function
def main(category,input_data):
    if category =='A':

        location,date=parse_weather_query(input_data)

        # error checker
        if not location:
            ans = (f"❌ Not found valid location，Please search the following location：{', '.join(locations)}")
            return ans
        if not date:
            ans = ("❌ Not found valid date，please use yyyy/mm/dd or yyyy.mm.dd format（for example 2024/01/05）Or 'tomorrow', 'today', 'next Monday' 。")
            return ans

        temperature,humidity=pred_a(location,date)
        ans =  "".join(["in ", location, " at date:", date, " is :", str(temperature), "℃ ", str(float(temperature) * 9/5 + 32), " ℉ ", "The humidity is :", str(humidity), "%"])
        return ans

    elif category =='B':
        date = extract_date(input_data)

        if not date:
            ans = (
                "❌ Not found valid date，please use yyyy/mm/dd or yyyy.mm.dd format（for example 2024/01/05）Or 'tomorrow', 'today', 'next Monday' 。")
            return ans

        ans = round(float(pred_b(date)),2)
        ans ="".join([ "at date ",date," is: ",str(ans)," AQI (air quality index)"])
        return ans

    elif category =='C':
        date = extract_date(input_data)

        if not date:
            ans = ("❌ Not found valid date，please use yyyy/mm/dd or yyyy.mm.dd format（for example 2024/01/05）Or 'tomorrow', 'today', 'next Monday' 。")
            return ans

        price,percent = pred_c(date)
        ans = "".join(["At date ", date, ", the prediction of S&P 500 Price is : ", str(price),", The Rise&Fall rate is : ", str(percent),"%"])
        return ans

def print_result(category,input,output):
    if category =='A':
        result = f"The weather {output}"
    elif category =='B':
        result = f"The air quality {output}"
    elif category =='C':
        result = f"{output}"
    return result


# Nature Language time process & match（into yyyy/mm/dd）
def extract_date(text):
    today = datetime.today()
    match = re.search(date_pattern, text)
    if match:
        return f"{match.group(1)}/{match.group(2).zfill(2)}/{match.group(3).zfill(2)}"
    text = re.sub(r"\s+", "", text.strip().lower())
    # process NL date
    if "tomorrow" in text.lower():
        return (today + timedelta(days=1)).strftime("%Y/%m/%d")
    if "today" in text.lower():
        return today.strftime("%Y/%m/%d")
    if "nextmonday" in text:
        days = 7 - today.weekday()
        return (today + timedelta(days=days)).strftime("%Y/%m/%d")
    if "nexttuesday" in text:
        days = 8 - today.weekday()
        return (today + timedelta(days=days)).strftime("%Y/%m/%d")
    if "nextwednesday" in text:
        days = 9- today.weekday()
        return (today + timedelta(days=days)).strftime("%Y/%m/%d")
    if "nextthursday" in text:
        days = 10 - today.weekday()
        return (today + timedelta(days=days)).strftime("%Y/%m/%d")
    if "nextfriday" in text:
        days = 11 - today.weekday()
        return (today + timedelta(days=days)).strftime("%Y/%m/%d")
    if "nextsaturday" in text:
        days = 12 - today.weekday()
        return (today + timedelta(days=days)).strftime("%Y/%m/%d")
    if "nextsunday" in text:
        days = 13 - today.weekday()
        return (today + timedelta(days=days)).strftime("%Y/%m/%d")

    if "thismonday" in text:
        days_until_monday = (0 - today.weekday()) % 7
        return (today + timedelta(days=days_until_monday)).strftime("%Y/%m/%d")
    if "thistuesday" in text:
        days_until_monday = (1 - today.weekday()) % 7
        return (today + timedelta(days=days_until_monday)).strftime("%Y/%m/%d")
    if "thiswednesday" in text:
        days_until_monday = (2 - today.weekday()) % 7
        return (today + timedelta(days=days_until_monday)).strftime("%Y/%m/%d")
    if "thisthursday" in text:
        days_until_monday = (3 - today.weekday()) % 7
        return (today + timedelta(days=days_until_monday)).strftime("%Y/%m/%d")
    if "thisfriday" in text:
        days_until_monday = (4 - today.weekday()) % 7
        return (today + timedelta(days=days_until_monday)).strftime("%Y/%m/%d")
    if "thissaturday" in text:
        days_until_monday = (5 - today.weekday()) % 7
        return (today + timedelta(days=days_until_monday)).strftime("%Y/%m/%d")
    if "thissunday" in text:
        days_until_monday = (6 - today.weekday()) % 7
        return (today + timedelta(days=days_until_monday)).strftime("%Y/%m/%d")

    return None  # not found

# location matcher
def extract_location(text):
    for loc in locations:
        if loc in text:
            return loc
    return None

# recognizer of nl-46/106 def
def parse_weather_query(query):
    location = extract_location(query)
    date = extract_date(query)
    return location, date

