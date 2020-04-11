# COVID-19 Risk Factor

"There is a growing urgency for these approaches because of the rapid increase in coronavirus literature, making it difficult for the medical community to keep up."[[1]](https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge) 
So this project aims to first find out the sentences talking about the risk factor of the COVID-19 form a large number of coronavirus literatures , and then recognize the risk factors in these sentences.
Advisor: [Pengtao Xie](https://sites.google.com/site/pengtaoxie2008/)

## 0  Repo introduction

[biobert_covid19_LM.ipynb](https://github.com/Molv1659/COVID-19-Risk-Factor/blob/master/biobert_covid19_LM.ipynb) pre-trains the BioBERT model on the CORD-19 dataset.

[biobert_text_classifier_3.ipynb](https://github.com/Molv1659/COVID-19-Risk-Factor/blob/master/biobert_text_classifier_3.ipynb) fine-tunes the  pre-trained model to classify the positive sentences that contain the risk factor of COVID-19.

[risk_factor_NER.ipynb](https://github.com/Molv1659/COVID-19-Risk-Factor/blob/master/risk_factor_NER.ipynb) trains a BiLSTM + CRF model to recognize the risk factors in the positive sentences. 

[crf.py](https://github.com/Molv1659/COVID-19-Risk-Factor/blob/master/crf.py) is called in risk_factor_NER.ipynb

[risk_factors_QA.ipynb](https://github.com/Molv1659/COVID-19-Risk-Factor/blob/master/risk_factors_QA.ipynb) fine-tunes the  pre-trained model to QA. This is what we finally use to filter the positive sentences and recognize the risk factors in them. 

[covid-19_risk_factor_sentence.ipynb](https://github.com/Molv1659/COVID-19-Risk-Factor/blob/master/covid-19_risk_factor_sentence.ipynb) preprocesses the data from the CORD-19 dataset to different forms for LM model and for Sequense Classification and selects the sentences containing the risk factors we chose.



## 1  Pre-train the [BioBERT](https://github.com/dmis-lab/biobert) model on the [CORD-19](https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge) dataset

BioBERT is a biomedical language representation model designed for biomedical NLP tasks. BioBERT is based on BERT and pretrained on PubMed abstracts and PubMed Central full-text articles(PMC).

CORD-19 is short for COVID-19 Open Research Dataset, which contains 47,000 scholarly articles about COVID-19, SARS-CoV-2 and related coronaviruses such as SARS-CoV and MERS-CoV.  We use a total of 13202 papers in the dataset.

By pre-training the BioBERT model on the CORD-19 dataset, the model can learn infomation about COVID-19 and work better while being applied to downstream text mining tasks. For this project, downstream tasks include Sequence Classification and Question answering.

>  n_gpu: 4
>
> per_gpu_train_batch_size = 8
>
> total_optimization steps: 10000                               

## 2  Collect a list of risk factors by searching Google

For example, from this page [https://www.cdc.gov/coronavirus/2019-ncov/specific-groups/high-risk-complications.html](https://www.cdc.gov/coronavirus/2019-ncov/specific-groups/high-risk-complications.html) we can collect risk factors including ['older adults', 'heart disease', 'Diabetes', 'Lung disease'].

Finally, we collect these risk factors. 

|                                                     | Risk Factors |
| ----------------------------------------------------| ----------- |
| Kinds of people    |older adults, pregant women|
| Kinds of diseases  |lung disease, heart disease, cardiovascular disease, chronic kidney disease, coronary heart disease, liver disease, nervous system disease, chronic respiratory disease, coagulation dysfunction|
| Specific diseases  |diabetes, HIV, cancer, asthma, sepsis, hypertension, arrhythmia, myocardial infarction, pneumonia, chronic renal failure, dyspnea, high fever|
| Specific substances|neutrophilia, lymphocytopenia, methylprednisolone, leukocytosis, higher lactate dehydrogenase (LDH), plasma urea, serum creatinine, IL-6, low CD4 cell count, CD3 T-cell, CD8 T-cell, increased high-sensitivity cardiac troponin, increased high-sensitivity C-reactive protein, D-dimer, aspartate aminotransferase, alanine aminotransferase|
| Other              |medical resources, socioeconomics|

They are only a small part of the risk factors of COVID-19, our goal is to find all risk factors in the CORD-19 dataset, which can't be done by letting a doctor read all the papers and find them.

## 3  Choose and label sentences

There are 101468 sentences in CORD-19 dataset that contain the risk factors above. Among them, the number of sentences that directly contain 'covid',  'cov' or 'coronavirus' is 2170. There are three kinds of sentences in them:

- Sentences talking about the risk factor of COVID-19  (P)
- Sentences talking about the risk factor of SARS-CoV and MERS-CoV  (N1)
- Sentences talking about the risk factor of  coronaviruses but we don't know whether it's true about COVID-19  (N2)

After reading these sentences we found that sentences N1 are much more than sentences P and N2,  and chose 300 positive sentence from above. 

We also read some sentences that contain 'COIVD-19' or '2019-nCoV' and chose some sentences talking about COVID-19 but not risk factors (N3)

Also there are huge amounts of irrelevant sentences in the CORD-19 dataset (N4). 

P :  positive sentences

Ni :  the  i_th  kind of negtive sentences

And there are many risk factors we didn't choose in section 2.  So by directly reading some papers in CORD-19 we chose another 200 positve sentences. 

Finally we have 500 positive sentences, 1648 negative sentences  belonging to N1, N2 or N3. (in [selected_500sen.xlsx](https://github.com/Molv1659/COVID-19-Risk-Factor/blob/master/data/selected_500sen.xlsx))

## 4  Sequence classification and find all sentences talking about the risk factor of COVID-19 in CORD-19 dataset

We use the [BertForSequenceClassification](https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#bertforsequenceclassification), which is a BERT model  with a linear layer on top. The weight and bias of the BERT model is from  the pre-trained model in section 1. By adding a linear layer, we use all the infomation of the BERT model's output, but not only the output corresponding to [CLS].

The test set has 300 sentences: 100 P, 100 ( N1, N2 or N3), 100 N4. (in [data_test.xlsx](https://github.com/Molv1659/COVID-19-Risk-Factor/blob/master/data/data_test.xlsx)) 

We have trained 3 classifiers:

|              |                Data                | Accuracy on Val | Acurracy on Test |
| ------------ | :--------------------------------: | :-------------: | :--------------: |
| Classifier_1 |     400 P, 1500 (N1, N2 or N3)     |      0.97       |       0.73       |
| Classifier_2 |           400 P, 1000 N4           |      0.97       |       0.79       |
| Classifier_3 | 400 P,  500 (N1, N2 or N3), 500 N4 |      0.94       |       0.96       |

The size of the training data for 3 classifiers are in the table above. And we use 75% as the training set and 25% as the validation set.  The data for classifier_3 is in [data3.xlsx](https://github.com/Molv1659/COVID-19-Risk-Factor/blob/master/data/data3.xlsx).

Classifier 1 and 2 work good on val set and not good on test set. Beca usse there are some kinds of sentences they have never been trained on, the outcome on the test set is anticipated.

$\frac{2}{3}+\frac{1}{3}\times\frac{1}{2}≈0.83 $, which means if the classifier_1 correctly predict all sentences belong to P, N123 and half of the sentences belong to N4, it will get about 0.83 accuracy on test set. So does classifier_2. And the data of classifier_1 is more unbalanced and it get lower accuracy on test set than classifier_2. So the accuracy 0.73 and 0.79 is  anticipated and reasonable. 

Although the sample is a little unbalanced and the number of negative sentences is twice that of positive ones,  classifier_3 works well on both val and test set. So we didn't specifically deal with the imbalance problem.

Next we applied classifier_3 to all 1,048,575  sentences in CORD-19 dataset. The classifier chose 39150 positive sentences. (in [positive_sent.xlsx](https://github.com/Molv1659/COVID-19-Risk-Factor/blob/master/data/positive_sent.xlsx))There are 4 kinds of sentences among them:

- S1: good positive sentences

- S2: sentences that don't contain useful information about risk factor

- S3: sentences that talk about the danger and fast spread of COVID-19

- S4: sentences that don't talk about risk factors

- S5: sentences talking about risk factor but not of COVID-19

S1 and S2 are rightly classified but we only want S1. S3, S4 and S5 are wrongly classified. We can filter out the S2, S3, S4 in subsequent processing. But if a positive sentence doesn’t contain 'COVID-19', we cannot decide whether it belongs to S1 or S5.

## 5  By QA, filtering out part of the sentences and recognizing the risk factors in positive sentences

Next section will talk about why not use NER to recognize the risk factor.

We fine-tuned pre-trained model in section1 on SQuAD1.0 dataset. We train 2 epochs and get the following results:

> f1 = 88.34
>
> exact_match = 80.85

The input for the model is [CLS] query [SEP] text [SEP]. The model will find the answer of the query in the text.

We used 'What is the risk factor' to query all the positive sentences and filtered out the answers that meet any of the following conditions:

- length ≤ 3
- contain '[CLS]' or '[SEP]' or 'risk factor'
- contain 'covid' or '2019 - ncov' or 'sars'
- not contain [a-z]

Most answers less than three in length don't make senses and most risk factorsare longer than 3. For sentences that  don't contain enough information about the risk factors (S2 and S4), the model has difficulty finding answers in the sentence text and answer tends to contain  '[CLS]' or '[SEP]' or 'risk factor'. For sentences that talk about the danger and fast spread of COVID-19 (S3), the model tends to give answers contain 'covid' or '2019 - ncov' or 'sars'. There are few answers don't contain [a-z] and they don't make sense.

 So we filtered out these answers. Although by doing this we filtered out part of good positive sentences (S1), the number of positive sentences is still great and we only need to focus on the quality of them, that is whether they really contain useful information about the risk factor of COVID-19.  After the filtering, we get 20280 positive sentences with their risk factors. (in [positive_sent_QA_fine_ver.xlsx](https://github.com/Molv1659/COVID-19-Risk-Factor/blob/master/data/positive_sent_QA_fine_ver.xlsx)) 

Below are 2 examples from it.

| Sentence                                                     | Risk Factor |
| ------------------------------------------------------------ | ----------- |
| Detection of specific antibodies in  patients with fever can be a good distinction between COVID-19 and other  diseases, so as to be a complement to nucleic acid diagnosis to early  diagnosis of suspected cases. | fever       |
| Diabetes was also  associated with an increased risk of SMO among cases complicated by severe  respiratory disease. | diabetes    |

The length distribution of the answer string is as follows:



![答案长度分布图](https://github.com/Molv1659/COVID-19-Risk-Factor/raw/master/pic_for_readme/答案长度分布图.png)

72.7% of answers have a string length less than 50, and these answers were composed of noun phrases. Most of those answers are risk factors of high quality.

The remaining 27.3% of the answers whose length is longer than 50 are considered not to have high quality: some answers contain unnecessary information, some positive sentences are too complicated for the model to sum up a short answer.

![答案长度分布50内](https://github.com/Molv1659/COVID-19-Risk-Factor/raw/master/pic_for_readme/答案长度分布50内.png)

The length distribution of answers whose length is less than 50 is as above. The answer with a string length of 15 is the most numerous. Basically, answers close to 15 in length have a higher probability density.



![答案出现次数分布](https://github.com/Molv1659/COVID-19-Risk-Factor/raw/master/pic_for_readme/答案出现次数分布.png)

There are 82 kinds of risk factors that occur more than 10 times in CORD-19 data set. Most risk factors occur only once or twice.

[待改：risk factor的展示]

## Why not use NER?

The training data are sentences that contain the risk factors we chose in section 2.(in [risk_factor_sen.npy](https://github.com/Molv1659/COVID-19-Risk-Factor/blob/master/data/risk_factor_sen.npy)) We use nltk to tokenize those sentences and tag the tokens in Begin, Inside, Outside (BIO) form.  We get the BIO tag by comparing to the  risk factors we chose in section 2. We set the maximum number of words in a sentence to 70 and filtered out the unqualified sentences. Finally the training data contains 4668 sentences. Below is an example:

```
The third influenza study was an RCT that investigated the efficacy of the influenza vaccine among older adults in Japanese LTCFs.
('The', 'DT', 'O')
('third', 'JJ', 'O')
('influenza', 'NN', 'O')
('study', 'NN', 'O')
('was', 'VBD', 'O')
('an', 'DT', 'O')
('RCT', 'NNP', 'O')
('that', 'WDT', 'O')
('investigated', 'VBD', 'O')
('the', 'DT', 'O')
('efficacy', 'NN', 'O')
('of', 'IN', 'O')
('the', 'DT', 'O')
('influenza', 'JJ', 'O')
('vaccine', 'NN', 'O')
('among', 'IN', 'O')
('older', 'JJR', 'B')
('adults', 'NNS', 'I')
('in', 'IN', 'O')
('Japanese', 'JJ', 'O')
('LTCFs', 'NNP', 'O')
```

We used BiLSTM + CRF model to do the NER job.  First there is an embedding layer and the embedding_size is 50, next is a BiLSTM layer whose hidden_size is 32 and activation function is 'tanh' and  in the end there is a CRF layer. The model was trained on 3501 samples and validated on 1167 examples.

![NER训练](https://github.com/Molv1659/COVID-19-Risk-Factor/raw/master/pic_for_readme/NER训练.png)



The model was trained for 5 epoches. Both accuracy and val_accuracy reach 100%. Through the analysis of training and validation loss, we can know that the model has not been overfitted. The NER model works well on train and val set.

But when we applied the model to the positive sentences, we found it only recognizes these risk factors that we used to label the BIO_tag. Because there are limited kinds of entities(41 risk factors we chose in section 2) in the training data, the model will tend to learn to recognize by these specific words and thus cannot recognize new risk factors. 

We cannot label many other kinds of risk factor entities because it's too time-consuming to do it. (And this is exactly what and why we hope to let computers do for us people) So this problem with NER model cannot be solved.



