/*Topic:CHICAGO CRIME ANALYSIS*/
/*TEAM MEMBERS:
		Tanish Kotyankar
           	Keyur Patel
              	Kyle Eifler
Course: BIA-652 Multivariate Data Analysis
Term: Fall 2018*/






/*
COOKD= Cook’s  influence statistic
COVRATIO=standard influence of observation on covariance of betas
DFFITS=standard influence of observation on predicted value
H=leverage,
LCL=lower bound of a % confidence interval for an individual prediction. This includes the variance of the error, as well as the variance of the parameter estimates.
LCLM=lower bound of a % confidence interval for the expected value (mean) of the dependent variable
PREDICTED | P= predicted values
RESIDUAL | R= residuals, calculated as ACTUAL minus PREDICTED
RSTUDENT=a studentized residual with the current observation deleted
STDI=standard error of the individual predicted value
STDP= standard error of the mean predicted value
STDR=standard error of the residual
STUDENT=studentized residuals, which are the residuals divided by their standard errors
UCL= upper bound of a % confidence interval for an individual prediction
UCLM= upper bound of a % confidence interval for the expected value (mean) of the dependent variable
* Cook’s  statistic lies above the horizontal reference line at value 4/n *;
* DFFITS’ statistic is greater in magnitude than 2sqrt(n/p);
* Durbin watson around 2 *;
* VIF over 10 multicolinear **;


*/

/*Parts of the SAS program:-
1, Data Cleaning
2, Make Dummy variables for Categorical variables
3, Get frequency of  each thype of crime
4, Sampling the data
5, Making test and training datasets
6, Logistic Regression
7, Scoring Logistic Regression model with the test and vice-a-versa
8,Confusion Matrix*/

proc import datafile = 'C:/Users/patel/downloads/Crimes_-_2001_to_present.csv'
 out = Data
 dbms = CSV;
run;
/*sort data with date, descending*/
proc sort data = Data out = Data;
by descending Date;
run;

/*Split Date and time variables*/
data TestTime;
set Data;
Date_r=datepart(Date);
Time_r=timepart(Date);
format Date_r yymmdd10. Time_r time20.;
run;
/*Removing Missing variables*/
data Clean;
	set TestTime;
	if district = . or Location = '' or X_coordinate = . or Y_coordinate = .
	or Ward = . or Community_area = . or District = 21 or District = 31 then delete;
	run;
/*Calculating frequency for each type of crime*/
proc freq data = Clean order=freq;
	Tables Primary_type/ out= FreqCount outexpect sparse;
 weight District;
Run;

/*Subset for offenses based on Primary_Type*/
data Data_major_offenses;
	set Clean;
	if Primary_type = 'ASSAULT' OR Primary_type ='NARCOTICS' OR Primary_type ='ROBBERY'
    OR Primary_type ='THEFT' OR Primary_type ='MOTOR VEHICLE THEFT' OR Primary_type ='CRIMINAL DAMAGE' or Primary_type = 'HOMICIDE';
run;

/*Subset for offenses that have a handgun used based on description*/
data Data_gun_offenses;
	set Clean;
	if find(Description,'Handgun','i') ge 1;
run;

/*Outliers*/
proc freq data = Clean order=freq;
	Tables Primary_type/ out= FreqCount outexpect sparse;
 weight District;
Run;


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


/*Clean District for null values
data data_dummy;
	set data_dummy;
	if District_8=. then District_8=0;
run;
*/

/*Down-Sampling Data for Criminal Damage*/
Title "Sampling and Logistic for Criminal Damage Train to Test";
proc sort data =data_dummy out=data_dummy1;
	by CD_D;
run;
proc freq data=data_dummy;
	Tables CD_D;
run;
proc surveyselect data = data_dummy1 out = cd method = srs sampsize=(695147,695147) seed = 9876;
	strata CD_D;
run;
proc freq data=cd;
	tables CD_D;
run;
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
/*Scoring the Criminal Damage model with test dataset*/
proc logistic inmodel=sasuser.CD_Model;
   score data=test_cd out=Score_CD fitstat;
run;
/*Confusion Matrix for Criminal Damage*/
proc freq data=score_cd;
   table F_CD_D*I_CD_D/nopercent nocol nocum out=CellCounts_cd;
run;

data CellCounts_cd;
     set CellCounts_cd;
     Match=0;
     if F_CD_D=I_CD_D then Match=1;
run;
/*Mean accuracy*/
proc means data=CellCounts_cd mean;
	freq count;
    var Match;
run;
/*Down-Sampling Data for Narcotics*/
Title "Sampling and Logistic for Narcotics Train to Test";
proc sort data =data_dummy out=data_dummy2;
	by Narcotics_D;
run;
proc surveyselect data = data_dummy2 out = narc method = srs sampsize=(634385,634385) seed = 9876;
	strata Narcotics_D;
run;
proc freq data=narc;
	tables Narcotics_D;
run;
/*Subset Testing and training dataset*/
data training_narc /*validation*/ test_narc;
  set narc;
  _n_=rand('uniform');
  if _n_ le .5 then output training_narc;
  else output test_narc;
run;
/*Logistic Regression for Narcotics*/
ods graphics on;
Title "Logistic Regression for Narcotics";
proc logistic data=training_narc plots=effect outmodel=sasuser.narc_Model;
	class Narcotics_D(ref='0') District(ref='1') Arrest_D(ref='0') Domestic_D(ref='0') /param=ref ;
	model  Narcotics_D= District Arrest_D Domestic_D;
	output out=narc_out pred=C;
quit;
/*Scoring the Narcotics model with test dataset*/
proc logistic inmodel=sasuser.narc_Model;
   score data=test_narc out=Score_narc fitstat;
run;
/*Confusion Matrix for Narcotics*/
proc freq data=score_narc;
   table F_Narcotics_D*I_Narcotics_D/nopercent nocol nocum out=CellCounts_narc;
run;

data CellCounts_narc;
     set CellCounts_narc;
     Match=0;
     if F_Narcotics_D=I_Narcotics_D then Match=1;
run;
/*Mean accuracy for Narcotics*/
proc means data=CellCounts_narc mean;
    freq count;
    var Match;
run;

/*Down-Sampling Data for Robbery*/
Title "Sampling and Logistic for Robbery Train to Test";
proc sort data =data_dummy out=data_dummy3;
	by Robbery_D;
run;
proc surveyselect data = data_dummy3 out = rob method = srs sampsize=(230660,230660) seed = 9876;
	strata Robbery_D;
run;
proc freq data=rob;
	tables Robbery_D;
run;
/*Subset Testing and training dataset*/
data training_rob /*validation*/ test_rob;
  set rob;
  _n_=rand('uniform');
  if _n_ le .5 then output training_rob;
  else output test_rob;
run;


/*Logistic Regression for Robbery*/
ods graphics on;
Title "Logistic Regression for Robbery";
proc logistic data=training_rob plots=effect outmodel=sasuser.rob_Model;
	class Robbery_D(ref='0') District(ref='1') Arrest_D(ref='0') Domestic_D(ref='0') /param=ref ;
	model  Robbery_D= District Arrest_D Domestic_D;
	output out=rob_out pred=C;
quit;
/*Scoring Robbery model with test dataset*/
proc logistic inmodel=sasuser.rob_Model;
   score data=test_rob out=Score_rob fitstat;
run;
/*Confusion Matrix for Robbery*/
proc freq data=score_rob;
   table F_Robbery_D*I_Robbery_D/nopercent nocol nocum out=CellCounts_rob;
run;

data CellCounts_rob;
     set CellCounts_rob;
     Match=0;
     if F_Robbery_D=I_Robbery_D then Match=1;
run;
/*Mean accuracy for Robbery*/
proc means data=CellCounts_rob mean;
    freq count;
    var Match;
run;
/*Down-Sampling Data for Motor Vehicle Theft*/
Title "Sampling and Logistic for Motor Vehicle Theft Train to Test";
proc sort data =data_dummy out=data_dummy3;
	by MVT_D;
run;
proc freq data=data_dummy3;
	Tables MVT_D;
run;
proc surveyselect data = data_dummy3 out = mvt method = srs sampsize=(278115,278115) seed = 9876;
	strata MVT_D;
run;
proc freq data=mvt;
	tables MVT_D;
run;
/*Subset Testing and training dataset*/
data training_mvt /*validation*/ test_mvt;
  set mvt;
  _n_=rand('uniform');
  if _n_ le .5 then output training_mvt;
  else output test_mvt;
run;
/*Logistic Regression for Criminal Motor Vehicle Theft*/
ods graphics on;
Title "Logistic Regression for Motor Vehicle Theft";
proc logistic data=training_mvt plots=effect outmodel=sasuser.mvt_Model;
	class MVT_D(ref='0') District(ref='1') Arrest_D(ref='0') Domestic_D(ref='0') /param=ref ;
	model  MVT_D = District Arrest_D Domestic_D;
	output out=mvt_out pred=C;
quit;
/*Scoring Motor Vehicle Theft model with test dataset*/
proc logistic inmodel=sasuser.mvt_Model;
   score data=test_mvt out=Score_mvt fitstat;
run;
/*Confusion Matrix for Motor Vehicle Theft*/
proc freq data=score_mvt;
   table F_MVT_D*I_MVT_D/nopercent nocol nocum out=CellCounts_mvt;
run;

data CellCounts_mvt;
     set CellCounts_mvt;
     Match=0;
     if F_MVT_D=I_MVT_D then Match=1;
run;
/*Mean accuracy for Motor Vehicle Theft*/
proc means data=CellCounts_mvt mean;
	freq count;
    var Match;
run;
/*Down-Sampling Data for Assault*/
Title "Sampling and Logistic for Assault Train to Test";
proc sort data =data_dummy out=data_dummy4;
	by Assault_D;
run;
proc freq data=data_dummy4;
	Tables Assault_D;
run;
proc surveyselect data = data_dummy4 out = ass method = srs sampsize=(375968,375968) seed = 9876;
	strata Assault_D;
run;
proc freq data=ass;
	tables Assault_D;
run;
/*Subset Testing and training dataset*/
data training_ass /*validation*/ test_ass;
  set ass;
  _n_=rand('uniform');
  if _n_ le .5 then output training_ass;
  else output test_ass;
run;
/*Logistic Regression for Assault*/
ods graphics on;
Title "Logistic Regression for Assault";
proc logistic data=training_ass plots=effect outmodel=sasuser.ass_Model;
	class Assault_D(ref='0') District(ref='1') Arrest_D(ref='0') Domestic_D(ref='0') /param=ref ;
	model  Assault_D = District Arrest_D Domestic_D;
	output out=ass_out pred=C;
quit;
/*Scoring Assault model with test dataset*/
proc logistic inmodel=sasuser.ass_Model;
   score data=test_ass out=Score_ass fitstat;
run;
/*Confusion Matrix for Assault*/
proc freq data=score_ass;
   table F_Assault_D*I_Assault_D/nopercent nocol nocum out=CellCounts_ass;
run;

data CellCounts_ass;
     set CellCounts_ass;
     Match=0;
     if F_Assault_D=I_Assault_D then Match=1;
run;
/*Mean accuracy for Assault*/
proc means data=CellCounts_ass mean;
	freq count;
    var Match;
run;

data training /*validation*/ test;
  set Data_dummy;
  _n_=rand('uniform');
  if _n_ le .50 then output training;
  /*else if _n_ le .8 then output validation;*/
  else output test;
run;
/*Logistic Regression for Theft*/

ods graphics on;
Title "Logistic Regression for Theft Train to test";
proc logistic data=training plots=effect outmodel=sasuser.theft_Model;
	class Theft_D(ref='0') District(ref='1') Arrest_D(ref='0') Domestic_D(ref='0') /param=ref ;
	model  Theft_D = District Arrest_D Domestic_D;
	output out=theft_out pred=C;
quit;
/*Scoring Theft model with test dataset*/
proc logistic inmodel=sasuser.theft_Model;
   score data=test out=Score_theft fitstat;
run;
/*Confusion Matrix for Theft*/
proc freq data=score_theft;
   table F_Theft_D*I_Theft_D/nopercent nocol nocum out=CellCounts_theft;
run;

data CellCounts_theft;
     set CellCounts_theft;
     Match=0;
     if F_theft_D=I_theft_D then Match=1;
run;
/*Mean accuracy for Theft*/
proc means data=CellCounts_theft mean;
	freq count;
    var Match;
run;




/*#################################Running the model on test dataset and scoring it on training dataset######################################################3####################################################################*/
/*Training model on Test and scoring with Train*/
ods graphics on;
Title "Logistic Regression for Criminal Damage Test to Train";
proc logistic data=test_cd plots=effect outmodel=sasuser.CD_Model;
	class CD_D(ref='0') District(ref='1') Arrest_D(ref='0') Domestic_D(ref='0') /param=ref ;
	model  CD_D= District Arrest_D Domestic_D;
	output out=CD_out pred=C;
quit;

proc logistic inmodel=sasuser.CD_Model;
   score data=training_cd out=Score_CD fitstat;
run;
/*Confusion Matrix for Criminal Damage*/
proc freq data=score_cd;
   table F_CD_D*I_CD_D/nopercent nocol nocum out=CellCounts_cd;
run;

data CellCounts_cd;
     set CellCounts_cd;
     Match=0;
     if F_CD_D=I_CD_D then Match=1;
run;
proc means data=CellCounts_cd mean;
	freq count;
    var Match;
run;
/*Logistic Regression for Narcotics*/
ods graphics on;
Title "Logistic Regression for Narcotics Test to Train";
proc logistic data=test_narc plots=effect outmodel=sasuser.narc_Model;
	class Narcotics_D(ref='0') District(ref='1') Arrest_D(ref='0') Domestic_D(ref='0') /param=ref ;
	model  Narcotics_D= District Arrest_D Domestic_D;
	output out=narc_out pred=C;
quit;
/*Scoring Narcotics model with train dataset*/
proc logistic inmodel=sasuser.narc_Model;
   score data=training_narc out=Score_narc fitstat;
run;
/*Confusion Matrix for Narcotics*/
proc freq data=score_narc;
   table F_Narcotics_D*I_Narcotics_D/nopercent nocol nocum out=CellCounts_narc;
run;

data CellCounts_narc;
     set CellCounts_narc;
     Match=0;
     if F_Narcotics_D=I_Narcotics_D then Match=1;
run;
/*Mean accuracy for Narcotics*/
proc means data=CellCounts_narc mean;
    freq count;
    var Match;
run;

/*Logistic Regression for Robbery*/
ods graphics on;
Title "Logistic Regression for Robbery Test to Train";
proc logistic data=test_rob plots=effect outmodel=sasuser.rob_Model;
	class Robbery_D(ref='0') District(ref='1') Arrest_D(ref='0') Domestic_D(ref='0') /param=ref ;
	model  Robbery_D= District Arrest_D Domestic_D;
	output out=rob_out pred=C;
quit;
/*Scoring Robbery model with train dataset*/
proc logistic inmodel=sasuser.rob_Model;
   score data=training_rob out=Score_rob fitstat;
run;
/*Confusion Matrix for Robbery*/
proc freq data=score_rob;
   table F_Robbery_D*I_Robbery_D/nopercent nocol nocum out=CellCounts_rob;
run;

data CellCounts_rob;
     set CellCounts_rob;
     Match=0;
     if F_Robbery_D=I_Robbery_D then Match=1;
run;
/*Mean accuracy for Robbery*/
proc means data=CellCounts_rob mean;
    freq count;
    var Match;
run;
/*Logistic Regression for Motor Vehivle Theft*/
ods graphics on;
Title "Logistic Regression for Motor Vehicle Theft Test to Train";
proc logistic data=test_mvt plots=effect outmodel=sasuser.mvt_Model;
	class MVT_D(ref='0') District(ref='1') Arrest_D(ref='0') Domestic_D(ref='0') /param=ref ;
	model  MVT_D = District Arrest_D Domestic_D;
	output out=mvt_out pred=C;
quit;
/*Scoring Motor Vehicle Theft model with train dataset*/
proc logistic inmodel=sasuser.mvt_Model;
   score data=training_mvt out=Score_mvt fitstat;
run;
/*Confusion Matrix Motor Vehicle Theft*/
proc freq data=score_mvt;
   table F_MVT_D*I_MVT_D/nopercent nocol nocum out=CellCounts_mvt;
run;

data CellCounts_mvt;
     set CellCounts_mvt;
     Match=0;
     if F_MVT_D=I_MVT_D then Match=1;
run;
/*Mean accuracy for Motor Vehicle Theft*/
proc means data=CellCounts_mvt mean;
	freq count;
    var Match;
run;
/*Logistic Regression for Assault*/
ods graphics on;
Title "Logistic Regression for Assault Test to Train";
proc logistic data=test_ass plots=effect outmodel=sasuser.ass_Model;
	class Assault_D(ref='0') District(ref='1') Arrest_D(ref='0') Domestic_D(ref='0') /param=ref ;
	model  Assault_D = District Arrest_D Domestic_D;
	output out=ass_out pred=C;
quit;
/*Scoring Assault model with train dataset*/
proc logistic inmodel=sasuser.ass_Model;
   score data=training_ass out=Score_ass fitstat;
run;
/*Confusion Matrix for Assault*/
proc freq data=score_ass;
   table F_Assault_D*I_Assault_D/nopercent nocol nocum out=CellCounts_ass;
run;

data CellCounts_ass;
     set CellCounts_ass;
     Match=0;
     if F_Assault_D=I_Assault_D then Match=1;
run;
/*Mean accuracy for Assault*/
proc means data=CellCounts_ass mean;
	freq count;
    var Match;
run;

Title "Theft Test to Train";
/*Subset Testing and training dataset*/
data training_theft /*validation*/ test_theft;
  set data_dummy;
  _n_=rand('uniform');
  if _n_ le .5 then output training_theft;
  else output test_theft;
run;
/*Logistic Regression for Theft*/
ods graphics on;
Title "Logistic Regression for Theft Test to Train";
proc logistic data=test_theft plots=effect outmodel=sasuser.theft_Model;
	class theft_D(ref='0') District(ref='1') Arrest_D(ref='0') Domestic_D(ref='0') /param=ref ;
	model  theft_D = District Arrest_D Domestic_D;
	output out=theft_out pred=C;
quit;
/*Scoring Theft model with train dataset*/
proc logistic inmodel=sasuser.theft_Model;
   score data=training_theft out=Score_theft fitstat;
run;
/*Confusion Matrix for Theft*/
proc freq data=score_theft;
   table F_Theft_D*I_Theft_D/nopercent nocol nocum out=CellCounts_theft;
run;

data CellCounts_theft;
     set CellCounts_theft;
     Match=0;
     if F_theft_D=I_theft_D then Match=1;
run;
/*Mean accuracy for Theft*/
proc means data=CellCounts_theft mean;
	freq count;
    var Match;
run;
