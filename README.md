---
# Synthetic Data Experimental Research System

### Euan Gardner
### Senior Information Development Manager
---

### Welcome!
Thank you for your interest in the synthetic data system! It's currently in Alpha version so don't expect perfect performance but please don't hesitate to give any feedback, ideas or comments. We've released it in this form so that we can get as many ideas and issues resolved before formally releasing so thank you again for your interest.

The system was built around medical data and hence may not be suitable for all your needs, if you've small data or data with few categories then I would highly recommend looking at alternative synthetic methods - e.g. SMOTE. We wanted to release the system for transparency, feedback and that it might help people with data science as the methods used should translate to non-medical data.

### Access needs
If you have any access needs (such as large print), please don't hesitate to contact me!

phs.synthetic-data-phs@nhs.net

### Windows Support
We are trying our best to support the Windows build of the SDS. We are confident that the Linux build will run, however there are some issues we are working on before fully supporting the Windows build. If you can help in anyway with this or want to try it please do so, we welcome any feedback!

### What is this?
Synthetic data is very similar to the original real data in that it has the same categories and data types, but contains completely made up people/objects in the data. High quality synthetic data has the additional need to be statistically similar to the real data as well.

### What does it do?
The current system is designed to handle large amounts of data with minimal input from you so that you can synthesise complex data quickly! It's designed around a mixture of traditional probability sampling, machine learning and differential privacy methods.

  - Uses GPUs for massive speed boost for Decision Tree fitting
  - Automatically handles missing data
  - Removes any data below a count specified to increase privacy as small counts can be problematic
  - Can create synthetic labels for more accurate synthesis
  - Performs iterative synthesis
  - Cluster numeric data using Gaussian Mixture Models (GMM) to make synthesis easier
  - Uses probability sampling to handle demographic variables
  - Fits and returns data from Random Forest models
  - All parameters fully customisable but default parameters built in for ease of use

###### This system DOES NOT build in Differential Privacy - see Cautions and Notes Section

### How do I work this?
![alt text](https://github.com/SDS-Architect/Synthetic_Data_System/blob/master/SDS/flowchart%20for%20SD.png)




If you do use this system or talk about it then please cite it as:

##### Gardner, E. (2019). Synthetic Data Experimental Research System (Version 0.1a) [software]. Available from: https://github.com/SDS-Architect/SDS_Public


<br />

- - -
# Cautions and notes

**Please review the Open Government License agreement used with this software.**

**Please DO NOT rely on this system to release synthetic data after synthesis without checks as it is still under quality and privacy evaluation. Use the code, play around with it, evaluate it and give feedback.**

**NHS NSS and Public Health Scotland are in no way responsible for your use of this system, any action or consequences that arise from the use of this system, it's download or your interaction with it.**

**While the system code/documentation has statements that it builds in Differential Privacy, after research and advice this is not the case. Future updates will, therefore, remove all mentions of differential privacy - in line with openess/research good practises. To clarifiy again, the system DOES NOT build in differential privacy **

**The author has no control over the release of the synthetic medical data itself and does not have the authority to influence the release process in anyway. As such, the author requests that you do not ask questions around this topic. Any comments, feedback or suggestions are more than welcome, as are any discussions on data, but the author cannot comment on or influence the release of the synthetic medical data itself. **
 - - -
