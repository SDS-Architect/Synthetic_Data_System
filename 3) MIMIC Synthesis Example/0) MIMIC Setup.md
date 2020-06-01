# What is MIMIC?
MIMIC is an openly available large dataset that contains a lot of anonymised information on patients who were in an American hospital's Intensive Care Unit (ICU). The data contains everything from basic admission information (date/time etc.) through to things like diagnosis(es) and lab test results.

It is a great resource for data science in general and has an additional advantage in the context of the synthetic data system.

#### Full reference
MIMIC-III, a freely accessible critical care database. Johnson AEW, Pollard TJ, Shen L, Lehman L, Feng M, Ghassemi M, Moody B, Szolovits P, Celi LA, and Mark RG. Scientific Data (2016). DOI: 10.1038/sdata.2016.35. Available from: http://www.nature.com/articles/sdata201635

# Why use MIMIC?
There are two main reasons we chose MIMIC to demonstrate the synthetic data system. The first is that the dataset has a similar structure and information to the datasets held by the NHS - Scottish Morbidity Records 1 (SMR01), on inpatient activity. For example, MIMIC uses International Classification of Diseases version 9 (ICD-9) codes and descriptions for diagnoses.

SMR01 in its current form uses ICD-10 codes but, MIMIC is still a very representative dataset in terms of structure. The MIMIC dataset also has columns with many categories (high cardinality) that the synthetic data system is designed to handle.

The second reason is that the dataset is also open and available to anyone, pending a free course and test. This means that you can use similar data to what the NHS developed the system on. I want this system to be open and tested/improved by the


# How do I access MIMIC?
Go to: https://mimic.physionet.org/gettingstarted/access/ and follow the instructions. It can take a few days to access the data but it is worth it - not just because you can use this amazing, bug free, no issues, perfect 110% great synthetic experimental system*!

*Not in any way guaranteed to be actually amazing, bug free, issue free or generate perfect synthetic data.

<br />


