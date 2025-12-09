# AI-Based Project Risk Prediction System

This project focuses on building a Machine Learning-based system to predict the **Risk Level of projects** using historical project management data. The model classifies projects into four risk categories:

- Critical  
- High  
- Medium  
- Low  

The system is designed to support **Project Managers (PMs) and Project Management Offices (PMOs)** by providing **early risk warnings**, enabling proactive decision-making to avoid budget overruns, delays, and project failures.

--------------------------------------------------
DATASET
--------------------------------------------------

Dataset Source:
Project management Risk Raw (Kaggle)

What the Dataset Contains:
- Project Type  
- Team Size  
- Project Budget  
- Estimated Timeline  
- Complexity Score  
- Stakeholder Count  
- Methodology Used  
- Team Experience Level  
- Past Similar Projects  
- Change Request Frequency  
- Requirement Stability  
- Vendor Reliability  
- Technical Debt Level  
- Schedule Pressure  
- Resource Availability  
- Client Experience Level  
- Industry Volatility  
- Risk Management Maturity  
- Final Target Column: Risk_Level (Critical, High, Medium, Low)

Origin of Dataset:
The dataset is robustly designed collection of 50 simulated project management related data points suited for practicing Exploratory Data Analysis and machine learning.

--------------------------------------------------
PROJECT MOTIVATION
--------------------------------------------------

Organizations handle **multiple projects simultaneously**, and managing risk across all these projects is extremely challenging. Many factors such as:

- Team experience  
- Budget pressure  
- Schedule constraints  
- Technical complexity  
- Stakeholder involvement  

can directly influence the success or failure of a project.

The motivation behind this project is to:
- Identify **high-risk projects early**
- Help project managers **prioritize resources**
- Reduce **cost overruns and delays**
- Improve **overall project success rates**

--------------------------------------------------
EXPLORATORY DATA ANALYSIS (EDA)
--------------------------------------------------

The following key questions were answered during EDA:

1. What is the distribution of Risk Levels across projects?
2. How do complexity, schedule pressure, and technical debt impact Risk Level?
3. Which project methodologies are associated with higher risk?
4. How does team experience affect project risk?
5. What is the relationship between budget utilization and risk?

Visualizations Implemented:
- Risk Level Distribution
- Numerical Feature Distributions
- Outlier Detection
- Categorical Feature Distributions
- Feature vs Risk Level Analysis

All EDA was implemented using **histograms, boxplots, bar charts, and count plots**.

--------------------------------------------------
MODELING AND TRAINING
--------------------------------------------------

Target Variable:
- Risk_Level (Critical, High, Medium, Low)

Machine Learning Models Used:
1. Logistic Regression
2. Random Forest Classifier
3. Hierarchical Random Forest

Model Comparison:
- Logistic Regression achieved approximately **72% accuracy**
- Random Forest achieved moderate performance (~50–55%)
- Hierarchical Random Forest showed very high accuracy but was identified as **overfitting**

Why Logistic Regression Was Selected as the Final Model:
- Simple and interpretable
- Stable generalization on unseen data
- Lower risk of overfitting
- Suitable for structured tabular project data

Issues Faced:
- High number of categorical variables
- Class imbalance in Risk_Level
- Overfitting in complex ensemble models
- Feature dimensionality explosion due to One-Hot Encoding

How the Issues Were Solved:
- Used One-Hot Encoding with proper preprocessing pipelines
- Used class balancing in Random Forest
- Used Stratified Train-Test split
- Compared multiple models for fair evaluation

--------------------------------------------------
BUSINESS INTERPRETATION OF RESULTS
--------------------------------------------------

The model provides **early risk warnings before project failure occurs**.

Using this model, Project Managers can:
- Prioritize **Critical & High-risk projects**
- Allocate **senior resources early**
- Strengthen **risk mitigation strategies**
- Control **cost overruns and schedule delays**
- Improve **overall project success rate**

--------------------------------------------------
LIMITATIONS & ASSUMPTIONS
--------------------------------------------------

- Some real-world variables like political risk, legal risk, and economic shocks are not included
- The model performance depends heavily on **data quality**
- Hierarchical model showed signs of **overfitting**

--------------------------------------------------
FUTURE WORK
--------------------------------------------------

Potential Improvements:
- Use real industry project data
- Add time-based risk forecasting
- Deploy using cloud-based systems
- Integrate with Jira, SAP, or ERP tools
- Implement explainable AI tools like SHAP

Next Steps:
- Convert into a full-scale business dashboard
- Add real-time project monitoring
- Automate weekly risk reporting

--------------------------------------------------
AUTHOR DETAILS
--------------------------------------------------

This project was created by:

Manjunath Mallikarjun Kendhuli  
Course: Introduction to AI & Machine Learning  
University: Hochschule Fresenius, Germany  

If you have any questions or feedback, feel free to connect with me on LinkedIn.

--------------------------------------------------
