# CBD_Stone

Machine Learning-Based Decision-Making Tool for Improved Choledocholithiasis Prediction and Management

## Methods
A prediction model was created utilizing a gradient boosting machine algorithm to estimate the probability of CDL based on clinical presenting features. The model was trained using a cohort of patients presenting to a tertiary care hospital and affiliated community centers with symptoms suggestive of choledocholithiasis. The model was validated using a separate cohort of patients presenting to the same hospital system. The model was then used to create a decision-making tree to guide management of patients with suspected CDL. The model was compared to the 2019 American Society for Gastrointestinal Endoscopy (ASGE) guidelines for choledocholithiasis.

The GBM model demonstrated 86% accuracy and 85% precision in distinguishing CDL risk, with an AUC of 95%. The decision-making tree suggested MRCP for 14% of patients, ERCP for 36% of patients, EUS for 12% and CCY + IOC for 37% patients. This approach missed only 1% of common bile duct (CBD) stones (5/469 test set patients) by suggesting CCY + IOC. It assigned only 2.5% (12/469) patients without CBD stones to ERCP, compared to 13.8% using ASGE 2019 guidelines.
