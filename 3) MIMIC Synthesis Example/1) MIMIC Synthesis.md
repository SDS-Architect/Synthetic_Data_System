# I've got MIMIC, what now?
The guide below will take you through each part of the setup to run the system. (Currently Linux only!)

##### Running the Synthesis

1) Ensure your sds_env enviroment is activate (type: source activate sds_env, if you're in any doubt)

2) Navigate to the 3) MIMIC Synthesis Example Folder

3) Download and place the Admissions.csv file (from the MIMIC Website) into the folder

4) Enter: python mimic_create.py

5) When the dialog box opens, navigate to the Admissions.csv file and select it

6) Once the file has run, you should see a file called Ready_to_synth_MIMIC.csv

7) Enter: synthesise

8) When the dialog box opens, select the Ready_to_synth_MIMIC.csv file

9) The SDS will populate the terminal with some information about your system, please follow the instructions

10) You will now see a dialog box open up that will ask for information about the dataset. Please open either the MIMIC_Testing.txt or MIMIC_Testing_gpu.txt (depending on whether you want to use CPU or GPU parameters). Select and copy all from the .txt file of your choice and paste it into the dialog box.

11) If you want, change any of the parameters (e.g. number of synthetic rows) and when you are ready, press ctrl+s to start. This will trigger the synthesis process.

12) Go and make a cup of tea/coffee/choice of beverage and let the system work away!

13) Once it is finished you can close the terminal window

14) Your synthetic data should be called Synthetic_MIMIC_Admissions_Output.csv
