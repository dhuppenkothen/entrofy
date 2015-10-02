# entrofy
Participant selection for workshops and conferences made easy.

Selection of participants for meetings as a discrete optimization problem.

## Overview

Given a list of participants with various "attributes" (e.g., gender, career stage, subfield, geography, years since PhD), the code finds the distribution of values within each attribute (e.g, male, female) and generate a subset that approximates target value distributions (e.g, 50% male/female, 30% junior, 30% non-US).

Attributes and values are user determined and based on the data present in uploaded CSV file.

### Input 
- CSV file with candidates, attribute, and values.
- Target size of subset
- Target distribution of each value
- Weights assigned to each value 

### Output 
- Distribution of each value in input file
- Subset of candidates which approximate the target distributions
- Distribution of each value in subset

## Install Instructions

Download the app/ directory

```
pip install -r requirements.txt
python server.py
```
then go to http://localhost:5000/ in your web browser

## Usage Instructions
### Input CSV file format

- header row required. 
- first column required to be a unique identifier. Either a unique numerical kep (anonymous) or full name.
- each column is intrepreted as an attribute (e.g, gender) 
- each field in a column is intrepreted as a value of the attribute
- missing data should be left as empty strings

### Interface 

- to ignore a parameter, set the weight to 0
- pre-selected candidates may be selected (left-click) to ensure their inclusion in the subset

## Implementation Suggestions

- input list should be all acceptable candidates
- input list should include pre-selected candidates so they are included in the distributions

## Things you should know

- output list is not necessarily reproducible -- each run will create a new list
- numbers are assumed to be from a continous distribution (e.g., years since PhD)
