# retinal-disease-detection
predict retinal disease using mobilenetv2 model 
Description
The dataset is organized into 3 folders (train, test, val) and contains subfolders for each image category (NORMAL,CNV,DME,DRUSEN). There are 84,495 X-Ray images (JPEG) and 4 categories (NORMAL,CNV,DME,DRUSEN).

Images are labeled as (disease)-(randomized patient ID)-(image number by this patient) and split into 4 directories: CNV, DME, DRUSEN, and NORMAL.

Optical coherence tomography (OCT) images (Spectralis OCT, Heidelberg Engineering, Germany) were selected from retrospective cohorts of adult patients from the Shiley Eye Institute of the University of California San Diego, the California Retinal Research Foundation, Medical Center Ophthalmology Associates, the Shanghai First People’s Hospital, and Beijing Tongren Eye Center between July 1, 2013 and March 1, 2017.

Source / useful links
DataSource : https://www.kaggle.com/paultimothymooney/kermany2018
Citation : http://www.cell.com/cell/fulltext/S0092-8674(18)30154-5

Type of Machine Learning Problem
It is a one class classification problem, for a given image we need to predict if they are suffering from which disease.

Performance Metric
Metric(s):

Categorical Crossentropy
Confusion Matrix
How does it work ?
* Understanding image data * Image augmentation * Applying model
Conclusion
1. we applied image augmentation to this dataset.
2. we applied three model - InceptionNet, DenseNet, and ResNet.
3. we have taken the weight of every model which is trained on imagenet dataset, we have not freezed the layer because the retina dataset is different from imagenet dataset.
4. we used confusion matrix because dataset is imbalanced and so accuracy score may not give good sence of result.
5. By seeing the confusion matrix of all the three model we can say that inception net has best recall than other model, precision is also good for inception net.
