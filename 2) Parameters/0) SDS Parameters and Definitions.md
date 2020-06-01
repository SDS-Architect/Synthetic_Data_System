# Parameters Names and Use

This file is designed to give you an overview of the parameters in Launch.py that you can use to control your synthesis. Each variable will be described and some examples given. If you are looking for more developer style input, check out the Numpy Docstrings attached to each function.

This file is written in the context of the MIMIC example and will use the headings & data of it. 

## Order of Columns in your Data
The system synthesises from left to right and order doesn't play too much of an importance. It is recommended that you place your demographic vairables in any order and ML_vars columns in order of importance/highest number of categories, whichever you think is best. 


<br />

## Mandatory Parameters

### categorical_variables
These are variables that can only have discrete values in your data. Variables, for example, like occupation are categorical because you can either have an occupation or you can't. It doesn't make sense to say 0.735 times Occupation: Cleaner. If the values in a column in your data can be described like this then you should list them in the categorical_variables list. 

#### Example
categorical_variables = ['ADMITTIME',
                         'DISCHTIME',
                         'DEATHTIME',
                         'ADMISSION_TYPE',
                         'ADMISSION_LOCATION',
                         'DISCHARGE_LOCATION',
                         'INSURANCE',
                         'LANGUAGE',
                         'RELIGION',
                         'MARITAL_STATUS',
                         'ETHNICITY',
                         'EDREGTIME',
                         'EDOUTTIME',
                         'DIAGNOSIS',
                         'HOSPITAL_EXPIRE_FLAG',
                         'DEMO_GMM'
                         ]

<br />

### demographic_variables
The data in these columns tell you about the background information about the people/objects in your data. Things like location, occupation etc. are all examples of demographics. The reason the system asks for demographic variables separately is that they only take on a limited range of categories and are well understood probabilistically. 

Demographic variables are, therefore, better suited to traditional conditional probability sampling rather than elaborate modelling techniques. This also has the added benefit that the system can randomly add in Laplacian noise to distort the raw probabilities to add additional security around the information. 

#### Example
demographic_variables = ['ADMISSION_LOCATION',
                            'DISCHARGE_LOCATION',
                            'INSURANCE',
                            'LANGUAGE',
                            'RELIGION',
                            'MARITAL_STATUS',
                            'ETHNICITY']

<br />

### ML_vars
Once you've defined your demographic variables, any remaining ones will be synthesised using Random Forest methods. The synthesis works from left to right so it is recommended that you place your ML_vars columns in order of importance/highest number of categories, whichever you think is best. 

If your first ML_vars column has many categories then it might be best to create a synthetic label using the synth_label_cols & 
synth_label_cols_stucture parameters. More information on these can be found in the Optional Parameters section. 

Having just read my description, this is basically all columns minus the demographic ones. This is redundant so I'll patch it out in the next release. 

#### Example
ML_vars = ['ADMISSION_TYPE',
            'ADMITTIME',
            'DISCHTIME',
            'DEATHTIME',
            'HOSPITAL_EXPIRE_FLAG',
            'EDREGTIME',
            'EDOUTTIME',
            'HAS_CHARTEVENTS_DATA',
            'DIAGNOSIS',
            'DEMO_GMM'
             ]

<br />

### combination_cols
These variables are the key to getting good performance from your synthesis. The iterative synthesis process relies on combining a set of columns to create a new one that acts like an ID string. This new column then allows the programme to group collections of people/objects in the data into manageable chunks to be synthesised while still being fast. 

Choose 2 =< (or more depending on how complex your data is) of your ML_vars for here. The reason for this is that they will likely be the most important variables of interest. The system also uses these variables when grouped to loop over the data and remove small counts, with a default value of any group with a count of less than 10 to be dropped but this can be changed. 

#### Example

combination_cols = ['ADMISSION_LOCATION',
                    'DIAGNOSIS'
                    ]

<br />

### size_of_synth_rows
How many rows do you want in your final synthesised dataset? I would recommend picking an integer greater than your current file, it's 1980's shoulder pad rules, so bigger is better.  

Note that if you have 1000 different groups with a complex dataset and you only put down size_of_synth_rows = 10, don't be surprised if you're missing a few groups. 

#### Example
Real data size is 10000 rows

size_of_synth_rows = 100000

<br />

### GPU_IDs
The machine learning components of the system rely on Graphics Processing Units (GPUs) to massively increase the speed at which they can learn and synthesise data. You need a minimum of one. Your computer will automatically set everything up, you just need to put down the numbers of each GPU. 

#### Example

If you've 1 GPU then put: GPU_IDs = '0'

If you've 2 GPUs then put: GPU_IDs = '0: 1'

If you've more than 2 GPUs then know I am super jealous of you (just keep copying the above pattern). 

<br />

### name_of_output
The name of synthetic .csv file that the system will create. Note that you DO NOT need to add .csv to the end, just name it.

#### Example
name_of_output = "Heavy_Good_Synthetic_File_Best_One_Ever"

<br />

## Optional Parameters

### remove_small_vals
This will remove any row or group that is below the specified number. The reason for this function is that small count groups or datapoints can be disclosive so it is good to remove these. 

You need to balance this, as the higher the number is then the less data you have. That being said, noise is added in the system so this parameter this doesn't need to be overly large. 

#### Example
remove_small_vals = 5

<br />

### cutting_vars
If you want any data trimmed down in your data to either reduce the number of categories (remember the more categories you have then the more likely data will be removed by the threshold) or just clean it then you specify them here. 

### length_cuts
Controls how long the data in the above columns will be cut to.

#### Example
cutting_vars = ['ADMITTIME',
                 'DISCHTIME',
                 'DEATHTIME',
                 'EDREGTIME',
                 'EDOUTTIME'
                 ]

length_cuts = 2

<br />

### numeric_group_vars
If you have any numeric columns then these can be synthesised by including them in this parameter. The system will know to apply a Gaussian Mixture Model (GMM) that will automatically group the values for you (default number of groups is 10). At the end of the process the system will automatically sample from the appropriate distributions to get real values back

### number_gaussian
This controls how many different Gaussian distributions are allowed to be used (default = 10). The more you use then the more accurate the end data is but at a cost of compute time and power. 

### GMM_cutoff
The threshold at which if the number of missing strings in the columns that GMM is being applied to, is above then it will split the data to keep these. Otherwise if the number of Missing is below the threshold then it will just drop the data (default = 20).

#### Example
numeric_group_vars = ['DEMO_GMM']

<br />

### synth_label_cols
If you have complex data where there is quite a jump from your demographic variables to your first ML_vars column (i.e. there isn't a direct link) then you can create a synthetic label to help the system. You can have multiple synthetic columns, though I wouldn't recommend this unless you have many, many columns that have complex relationships between them. 

### synth_label_cols_structure
Related to above and controls what character(s) to cut from the data in the specified columns to create a synthetic label or labels. 

#### Example
synth_label_cols = ['ADMISSION_TYPE', 'DIAGNOSIS']

synth_label_cols_structure = [2, 4]

If you chose [-1, -1] instead then it would take the last character value, it's based on python indexing. 

<br />
