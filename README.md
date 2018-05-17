# Spam-Classification-with-SVM
Beginner machine learning project from data-processing to model training, still lot of scope of improvement!!

I would like to explain my approach for spam classification:
I this project I have used the dataset present on this https://spamassassin.apache.org/old/publiccorpus/. This dataset contains good amount spam and ham emails. As apart my experience everything depends on how you do the preprocess the data and extract features from it. As a part of this project I have extraced features from body of email, but we can also extract features from headers email like number of hops used to reach destination, valid Ip, proxy used and others can give us a good amount of information. 

Approach:

Preprocessing:
1) Each email contains lot of email addresses, http address so in each of the email we normailize this and replace it with some string like I used "httpaddr" and "emailaddr" for http and email  address respectively.
2) In this step we can identify thing as much as possible like "$some amount", zip codes, phone numbers, time, date, place and other things which are possible. All this things will make the email more descriptive and contain lot of information. In my approach I maily identified $some amount zipcodes this alone gave me 82% acurracy on my test after selection of best model using the cross-validation set, so you probably now know how impactful this thing can be. Let me give small motivation just inclusion of "$some amount" increased my accuracy by 8%.
3) Now in this I have removed the stopwords and all the punctuation and unnecessary info, which do not give much inforamtion. After this I used NLP tecnique stemming to normalize the words for example listening, listen,listened to listeni. I used potter steeming tool present In NLTK package for this.
4) Now, its time to create our feature i.e. which will be like the most frequent words accross all the mails. I have selected 800 words it can more also, but doing so will not much help if the processed email do not contain much information so do your preprocessing well!!!.

Model Traning:

I have use support vector machine as a machine learning algorithm, simply because of its outstanding robustness and it is a well known large margin classfier, which basically comes from the cost function we use. Now, going forward I have created a dataset of 800 most frequent words and the data set contains around 6K instances where 4K are non-spam and 2K are spam instances. 

While traning I have randomly suffle the data set and divided the data set into 70%-20%-10% split which is traning, cross-validation, test data. Finally, 

                  Accuracy 
Traning             99.93
Cross_validation    81.43
Test                82.63

This model can be further improved in great way!!! including more header features and more identifiactions or using sentence semantics features.


Thanks for tuning in!!! Hope you like it give your suggestions or new ideas!!!
