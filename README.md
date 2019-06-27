
For our coursework of Multivariate Analysis, our Professor Khasha Dehnad suggested that we analyze crimes in United States. After looking at state government websites we found a dataset for the City of Chicago which had Reported Crimes from 2001 to Present. The dataset is available at this [link](https://catalog.data.gov/dataset/crimes-2001-to-present-398a4).

The dataset has over 6.7 million rows and 22 variables. Since the dataset only has Categorical variables we couldn't apply any linear machine learning application to predict. We decided to appply Logistic Regression for this problem to find out the probability of a crime occurring.

Influential variables includes

Primary_type:- Its the type of crime being reported. For eg. Theft, Robbery, Assault

Arrest :- A boolean value for which an Arrest is made for the reported crimes

Domestic :- A boolean value for if the reported crime is a domestic offense or not

District(Target Variable) :- Categorical value which corresponds to the police district in Chicago

**Data Pre-Processing**

We subset part of the data and analyze it on Excel. We notice that there are some null values in district, Location, X_coordinate, Y_coordinate, Ward, Community_area and District 21 & 31 have less than 200 values which make it useless for this problem. We remove these null values in SAS

    ```
    data Clean;
    	set TestTime;
    	if district = . or Location = '' or X_coordinate = . or Y_coordinate = .
    	or Ward = . or Community_area = . or District = 21 or District = 31 then delete;
    run;

    ```

  The Date variable also has the time part in it. We split that variable to separate Date and Time.

      ```
      data TestTime;
        set Data;
        Date_r=datepart(Date);
        Time_r=timepart(Date);
        format Date_r yymmdd10. Time_r time20.;
      run;
      ```

   We then subset the dataset for Major Offenses. We picked the most recurring crimes.

   ```
   data Data_major_offenses;
   	set Clean;
	  if Primary_type = 'ASSAULT' OR Primary_type ='NARCOTICS OR Primary_type ='ROBBERY' OR Primary_type ='THEFT' OR Primary_type ='MOTOR VEHICLE THEFT' OR Primary_type ='CRIMINAL DAMAGE' OR Primary_type = 'HOMICIDE';
   run;
   ```

   To work with categorical data we need to convert them into dummy variables.

   ```
   /*Creating Dummy variables for District_1-Dictrict_25*/
   DATA Data_Dummy;
     set Data_major_offenses;

     ARRAY dummys {*} 3.  District_1 - District_25;
     DO i=1 TO 25;			      
       dummys(i) = 0;
     END;
     dummys(District) = 1;		
   RUN;

   /*Create Dummy variables for Primary_Type, Arrest and Domestic*/
   data data_dummy;
   	set data_dummy;
   	if Primary_type = 'ASSAULT' then Assault_D = 1;
   	else Assault_D = 0;

   	if Primary_type = 'NARCOTICS' then Narcotics_D = 1;
   	else Narcotics_D = 0;

   	if Primary_type = 'THEFT' then Theft_D = 1;
   	else Theft_D = 0;

   	if Primary_type = 'HOMICIDE' then Homicide_D = 1;
   	else Homicide_D = 0;

   	if Primary_type = 'CRIMINAL DAMAGE' then CD_D = 1;
   	else CD_D = 0;

   	if Primary_type = 'ROBBERY' then Robbery_D = 1;
   	else Robbery_D = 0;

   	if Primary_type = 'MOTOR VEHICLE THEFT' then MVT_D = 1;
   	else MVT_D = 0;

   	if Arrest = "true" then Arrest_D=1;
   	else Arrest_D=0;

   	if Domestic = "true" then Domestic_D=1;
   	else Domestic_D=0;

   	run;

   ```
We then moved onto analyzing the data. Since the dataset is huge and we decided to work on subsets of the data to the full dataset, we had a doubt that this would be an imbalanced class problem.

To confirm our suspicion we run **proc freq** on Criminal Damage

| CD_D | Frequency | Percent | CumulativeFrequency | CumulativePercent |
|------|-----------|---------|---------------------|-------------------|
| 0    | 2805450   | 80.14   | 2805450             | 80.14             |
| 1    | 695147    | 19.86   | 3500597             | 100.00            |

From the 80-20 ratio we can confirm our suspicions.

Application of any machine learning algorithms will give us an output with very high accuracy, which we know is biased towards 0.
To counter this, we need to apply sampling technique. We used Downsampling in SAS.

  ```
  proc surveyselect data = data_dummy1 out = cd method = srs sampsize=(695147,695147) seed = 9876;
  	strata CD_D;
  run;

  /*Output of Downsampling*/
  proc freq data=cd;
  	tables CD_D;
  run;
  ```

**Output**

  | CD_D | Frequency | Percent | CumulativeFrequency | CumulativePercent |
|------|-----------|---------|---------------------|-------------------|
| 0    | 695147    | 50.00   | 695147              | 50.00             |
| 1    | 695147    | 50.00   | 1390294             | 100.00            |

From the output we can see that the subset has been successfully downsampled and we can go on to apply Logistic Regression.

We split our sampled dataset into training and testing, with a split of 75-25. We then model Logistic Regression for Criminal Damage.

  ```
  /*Subset Testing and training dataset*/
  data training_cd /*validation*/ test_cd;
    set cd;
    _n_=rand('uniform');
    if _n_ le .75 then output training_cd;
    else output test_cd;
  run;

  /*Logistic Regression for Criminal Damage*/
  ods graphics on;
  Title "Logistic Regression for Criminal Damage";
  proc logistic data=training_cd plots=effect outmodel=sasuser.CD_Model;
  	class CD_D(ref='0') District(ref='1') Arrest_D(ref='0') Domestic_D(ref='0')/param=ref ;
  	model  CD_D= District Arrest_D Domestic_D;
  	output out=CD_out pred=C;
  quit;
  ```

**Output**

| Analysis of Maximum Likelihood  Estimates |    |    |          |               |                |            |
|:-----------------------------------------:|:--:|:--:|:--------:|:-------------:|:--------------:|:----------:|
| Parameter                                 |    | DF | Estimate | StandardError | WaldChi-Square | Pr > ChiSq |
| Intercept                                 |    | 1  | -0.7201  | 0.0123        | 3422.9869      | <.0001     |
| District                                  | 2  | 1  | 0.8821   | 0.0158        | 3109.6847      | <.0001     |
| District                                  | 3  | 1  | 1.1333   | 0.0156        | 5296.4252      | <.0001     |
| District                                  | 4  | 1  | 1.2103   | 0.0150        | 6494.7987      | <.0001     |
| District                                  | 5  | 1  | 1.2010   | 0.0159        | 5740.7155      | <.0001     |
| District                                  | 6  | 1  | 0.9563   | 0.0152        | 3980.6153      | <.0001     |
| District                                  | 7  | 1  | 1.1024   | 0.0152        | 5270.5673      | <.0001     |
| District                                  | 8  | 1  | 1.3417   | 0.0145        | 8600.8085      | <.0001     |
| District                                  | 9  | 1  | 1.2466   | 0.0154        | 6591.0456      | <.0001     |
| District                                  | 10 | 1  | 1.0404   | 0.0160        | 4212.3552      | <.0001     |
| District                                  | 11 | 1  | 0.7827   | 0.0152        | 2637.6068      | <.0001     |
| District                                  | 12 | 1  | 0.8881   | 0.0152        | 3406.1857      | <.0001     |
| District                                  | 14 | 1  | 0.9570   | 0.0158        | 3646.2335      | <.0001     |
| District                                  | 15 | 1  | 0.8787   | 0.0164        | 2861.1953      | <.0001     |
| District                                  | 16 | 1  | 1.3646   | 0.0164        | 6904.8120      | <.0001     |
| District                                  | 17 | 1  | 1.2239   | 0.0168        | 5278.6337      | <.0001     |
| District                                  | 18 | 1  | 0.3236   | 0.0161        | 402.3639       | <.0001     |
| District                                  | 19 | 1  | 0.7937   | 0.0156        | 2587.7183      | <.0001     |
| District                                  | 20 | 1  | 1.0785   | 0.0199        | 2928.9120      | <.0001     |
| District                                  | 22 | 1  | 1.1382   | 0.0166        | 4686.0748      | <.0001     |
| District                                  | 24 | 1  | 1.1738   | 0.0170        | 4785.6056      | <.0001     |
| District                                  | 25 | 1  | 1.0882   | 0.0150        | 5242.4116      | <.0001     |
| Arrest_D                                  | 1  | 1  | -1.8644  | 0.00629       | 87960.2646     | <.0001     |
| Domestic_D                                | 1  | 1  | 0.5647   | 0.00921       | 3757.6669      | <.0001     |

From the **Pr>ChiSq** values we can see that all the variables selected for the model are influential. This gave us hope that we were on the right path and our model should give us a good output.

We then score the model against the test dataset and create a confusion matrix to check the accuracy of our model.

  ```
  /*Scoring the Criminal Damage model with test dataset*/
  proc logistic inmodel=sasuser.CD_Model;
     score data=test_cd out=Score_CD fitstat;
  run;
  /*Confusion Matrix for Criminal Damage*/
  proc freq data=score_cd;
     table F_CD_D*I_CD_D/nopercent nocol nocum out=CellCounts_cd;
  run;

  ```

**Output**

| F_CD_D(From: CD_D) | I_CD_D(Into: CD_D) |        |        |
|:------------------:|:------------------:|--------|--------|
|                    | 0                  | 1      | Total  |
| 0                  | 141441             | 207255 | 348696 |
| 1                  | 39099              | 307690 | 346789 |
| Total              | 180540             | 514945 | 695485 |
|                    | 25.96%             | 74.04% | 100%   |

The confusion matrix gives us an accuracy of **64.637%**.

We then apply the same code to each and every Primary_type. We have posted the results on this Word file.

One of the most influential results that we found was for that of Narcotics. Since Chicago's Police Department crack down on Narcotics has seen them make numerous arrests, it also significantly improves the accuracy of our model.

| F_Narcotics_D(From: | I_Narcotics_D(Into: Narcotics_D) |        |        |
|---------------------|----------------------------------|--------|--------|
| Narcotics_D)        |                                  |        |        |
|                     | 0                                | 1      | Total  |
| 0                   | 283426                           | 34011  | 317437 |
| 1                   | 2020                             | 314765 | 316785 |
| Total               | 285446                           | 348776 | 634222 |
|                     | 45.01                            | 54.99  | 100.00 |

Which gave us an accuracy of **94.31%**

The result gave us a hint that the model may be biased towards just positives and considering that this was an imbalanced classification problem accuracy wouldn't provide us with a good measure. After some research we concluded that AUC would perhaps act as the best measure.

To test our skills and make it easier for us, we decided to use Python for this part of the problem. Just as before we sampled the dataset.

**Sampling the dataset in Python for Primary_Type = Criminal Damage**
```
#Keeping the required variables for Criminal Damage
df_cd = df[['CD_D', 'Arrest_D', 'Domestic_D', 'District_1', 'District_2', 'District_3', 'District_4', 'District_5', 'District_6', 'District_7', 'District_8', 'District_9',  'District_10', 'District_11', 'District_12', 'District_13', 'District_14', 'District_15', 'District_16', 'District_17', 'District_18', 'District_19', 'District_20', 'District_22', 'District_23', 'District_24', 'District_25']]

```
We split the dataset into train and test and define two functions,
  1. Logistic Regression with auc
  2. Random forest with auc

The following code shows the function defined for Logistic Regression
```
#Logistic Regression
def logistic(x_train,y_train,x_test,y_test,x):
    logistic = LogisticRegression().fit(x_train, y_train)
    pred = logistic.predict(x_test)
    print(confusion_matrix(y_test,pred))
    pred_log_prob = logistic.predict_proba(x_test)
    precision,recall, thresholds = precision_recall_curve(y_test,pred_log_prob[:,1])
    print("AUC Score for Logistic Regression for {} is {}".format(df_names[x],(auc(recall,precision))))
    pyplot.plot([0, 1], [0.5, 0.5], linestyle='--')
    pyplot.plot(recall, precision, marker='.')
    pyplot.show()

```

Result for Assault is

|   | 0      | 1      |
|---|--------|--------|
| 0 | 627077 | 153724 |
| 1 | 454672 | 326842 |

The AUC curve can be seen
[here](https://github.com/KeyurPatel0124/Chicago-Crime-Analysis/blob/Tanish/Results/Logistic%20Assault.png)



AUC score for Logistic Regression is **0.67976**

The following code shows the function defined for Random forest

```
def RandomForest(x_train,y_train,x_test,y_test,x):
  rf = RandomForestClassifier()
  RF_1 = rf.fit(x_train, y_train)
  pred_RF = RF_1.predict(x_test)
  #print(accuracy_score(y_test, pred_RF))
  print(confusion_matrix(y_test, pred_RF))
  #recall=recall_score(y_test, pred_RF)
  #precision=precision_score(y_test, pred_RF)
  pred_RF_prob = RF_1.predict_proba(x_test)
  precision,recall, thresholds = precision_recall_curve(y_test,pred_RF_prob[:,1])
  print("AUC Score for Random Forest for {} is {}".format(df_names[x],(auc(recall,precision))))
  pyplot.plot([0, 1], [0.5, 0.5], linestyle='--')
  pyplot.plot(recall, precision, marker='.')
  pyplot.show()

```

Result for Assault is

|   | 0      | 1      |
|---|--------|--------|
| 0 | 541913 | 238888 |
| 1 | 357039 | 424475 |

The AUC curve can be seen
[here](https://github.com/KeyurPatel0124/Chicago-Crime-Analysis/blob/Tanish/Results/Random%20Forest%20Assault.png)

AUC score for Random Forest is **0.69577**
