import numpy as np
import pandas as pd
import cv2
import re
import easyocr
from datetime import datetime


def extract_text_easyocr(image):
    # Use EasyOCR as an alternative
    reader = easyocr.Reader(['en'])
    result = reader.readtext(image, detail=0)  # Pass the preprocessed image here
    return " ".join(result)  # Join results into a single string


def extract_expiry_date(texts):
    expiry_dates = []
    for text in texts:
        # Regex for expiry formats
        date_patterns = [
            r'(\d{1,2}/\d{1,2}/\d{4})',  # DD/MM/YYYY
            r'(\d{1,2}/\d{1,2}/\d{2})',   # DD/MM/YY
            r'(\d{1,2}/\d{4})',           # MM/YYYY
            r'(\d{1,2}/\d{2})',           # MM/YY
            r'BEST BEFORE (\d+) MONTHS',  # BEST BEFORE X MONTHS
            r'BEST BEFORE (\d+) YEARS'    # BEST BEFORE X YEARS
        ]

        for pattern in date_patterns:
            matches = re.findall(pattern, text)
            if matches:
                expiry_dates.extend(matches)

    return expiry_dates


def extract_mrp(text):
    # Updated regex patterns for extracting MRP values
    mrp_patterns = [
        r'MRP\s*Rs\.?\s*[:\-]?\s*(\d+(?:\.\d{1,2})?)\s*(?:incl\.? of all taxes)?',  # MRP Rs: XX.XX
        r'MRP\s*[:\-]?\s*(\d+(?:\.\d{1,2})?)\s*(incl\.? of all taxes)?',  # MRP: XX.XX (incl. of all taxes)
        r'Sale Price\s*Rs\.?\s*[:\-]?\s*(\d+(?:\.\d{1,2})?)\s*(?:/|-)?',  # Sale Price: Rs. XX.XX
        r'Price\s*[:\-]?\s*Rs\.?\s*(\d+(?:\.\d{1,2})?)\s*(?:/|-)?',  # Price: Rs. XX.XX
        r'₹\s*(\d+(?:\.\d{1,2})?)\s*(?:/|-)?',  # ₹ XX.XX
    ]

    mrp_values = []
    for pattern in mrp_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)  # Ignore case for pattern matching
        if matches:
            # Extract MRP values, ensuring we only take valid floats
            for match in matches:
                if match:  # Ensure that the match is not empty
                    mrp_values.append(match)  # Add only the captured group

    # Remove duplicates
    mrp_values = list(set(mrp_values))  # Remove duplicates

    # Debugging: print the matched MRP values
    #print("Matched MRP Values:", mrp_values)

    return mrp_values


def filter_expiry_dates(expiry_dates):
    current_date = datetime.now()
    manf_dates = []
    expiry_date = []

    for date in expiry_dates:
        # Handle different date formats
        if '/' in date:  # Check if the date format is in DD/MM/YYYY or MM/YYYY or DD/MM
            parts = date.split('/')
            if len(parts) == 3:  # DD/MM/YYYY
                day, month, year = map(int, parts)
                if year < 100:  # If the year is two digits
                    year += 2000  # Convert to four-digit year
                parsed_date = datetime(year, month, day)  # Assume the first day of the month
            elif len(parts) == 2:  # MM/YYYY or MM/YY
                month, year = map(int, parts)
                if month > 0 and month < 13:
                    if year < 100:  # If the year is two digits
                        year += 2000  # Convert to four-digit year
                    parsed_date = datetime(year, month, 1)  # Assume the first day of the month
                else:
                    continue
            elif len(parts) == 2 and len(parts[1]) == 2:  # DD/MM/YY
                day, year = map(int, parts)
                year += 2000  # Convert to four-digit year
                parsed_date = datetime(year, day, 1)  # Assume the first day of the month
            else:
                continue  # Skip if the format does not match expected patterns

            # Compare with the current date
            if parsed_date > current_date:
                expiry_date.append(date)
            elif parsed_date <= current_date:
                manf_dates.append(date)

        # Check for "BEST BEFORE" phrases
        if 'BEST BEFORE' in date:
            months = re.findall(r'(\d+)', date)
            if months:
                months = int(months[0])
                expiry_date = manf_dates[0].replace(day=1) + pd.DateOffset(months=months)
                if expiry_date >= current_date:
                    return expiry_date  # Store the original date string

                else:
                    return None

    return expiry_date, manf_dates


def get_Text_exp_manf_mrp(img_path):
    # Update with your actual image path
    image_path = img_path

    # Load the image
    image = cv2.imread(image_path)

    # Use EasyOCR for extraction
    easyocr_text = extract_text_easyocr(image)  # Pass preprocessed image
    #print(easyocr_text)
    # Find expiry date in EasyOCR text
    expiry_dates = extract_expiry_date([easyocr_text])  # Wrap in a list

    # Extract MRP from the text
    mrp_values = extract_mrp(easyocr_text)
    print("MRP:", mrp_values)

    # Filter valid and expired dates
    expiry_date, manf_dates = filter_expiry_dates(expiry_dates)
    print(f"Manufacturing Dates: {manf_dates}")
    print(f"Expiry Dates: {expiry_date}")

    if not expiry_date or len(manf_dates) > 1:
        expiry = "EXPIRED"
    else:
        expiry = "NOT EXPIRED"
    print(f"Product is {expiry}")

get_Text_exp_manf_mrp("test.jpg")