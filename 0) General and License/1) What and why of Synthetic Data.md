# All about Synthetic Data and the Project

## Who Are We
We’re a team based at Public Health Scotland, Scotland’s lead agency for improving and protecting the health and wellbeing of all of Scotland’s people. The Synthetic data project aims to make realistic, but fake, individual level data more readily available to students and researchers whilst preserving people’s right to privacy.

## What is Synthetic Data?
Synthetic data is very similar to the original data in that it has the same variables, categories, and data types, but contains completely made up people/objects in the data. High quality synthetic data are also statistically similar to the real data so that if 3% of the real people have a particular condition, around 3% of the fake records in the Synthetic data will too.

## Why develop Synthetic data?
We are developing synthetic data to allow Scottish hospital activity data to be used to facilitate learning and innovation, while balancing this with people’s right to privacy. Synthetic data will enable us to provide access to high quality fake patient level data, while avoiding the use of real data.

As Artificial Intelligence (AI), Machine Learning (ML), Data Science and statistics increase in use then the problem of high quality and representative data can become an issue. This is to say that data contain a large imbalance of people, with one large group of people and other small groups of people. 

The current system aims to also be able to create more examples of smaller groups, to potentially rebalance a dataset. Please note that this isn't a panacea for all problems and that good data science practises and examination of the data/output will always be needed. This is simply an additional tool that may help.

## Is it easy to develop Synthetic Data?
Why is it so difficult to synthesise the data, don't we have tons of stuff that can do this? You're right, there are excellent open-source systems that have been used to synthesise things like census data. The problem comes in the form of the number of categories, formally known as cardinality. 

Current open-source systems are not designed to handle a large number of columns with many categories, which is what most hospital activity data contain. As an example, if you take the main diagnosis column of a hospital admission dataset and reduce it to just the letter and two numbers to keep it still clinically relevant, you still have close to a 1000 categories; e.g. A00 - Cholera. There are multiple columns for diagnoses with these categories and when you add operation codes, this becomes a tricky problem.

## So you've cracked the problem?
Initial signs are promising but there is still some way to go. The system appears to output quality synthetic data but PHS are currently running their own internal evaluation on the privacy and quality aspects of the synthetic data.


## Caveats
This system is still under evaluation. Use the code, play around with it, evaluate it and give feedback but PHS is in no way responsible for your use of this system. We strongly recommend that if you plan on releasing any synthetic data, that it be fully checked in your organisation before being released. 

The system is currently not designed to work on purely numeric or time-series data but these are future updates. All future update plans will be listed in the document: Feature Roadmap and will be updated as we receive more feedback.


