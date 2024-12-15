# Retinal-OCT-Images
to detect and classify human diseases from medical images.

<h2> Description </h2>

The dataset is organized into 3 folders (train, test, val) and contains subfolders for each image category (NORMAL,CNV,DME,DRUSEN). There are 84,495 X-Ray images (JPEG) and 4 categories (NORMAL,CNV,DME,DRUSEN).

Images are labeled as (disease)-(randomized patient ID)-(image number by this patient) and split into 4 directories: CNV, DME, DRUSEN, and NORMAL.

Optical Coherence Tomography (OCT) images are high-resolution cross-sectional images of the retina, commonly used for diagnosing eye conditions like Diabetic Macular Edema (DME), Choroidal Neovascularization (CNV), and Drusen. In this project, OCT images are classified into four categories: CNV, DME, DRUSEN, and NORMAL. The model leverages deep learning, specifically MobileNetV2, to accurately classify these images, helping in early diagnosis and treatment planning for retinal diseases.

<h2> Source / useful links </h2>

DataSource : https://www.kaggle.com/paultimothymooney/kermany2018 <br>
Citation : http://www.cell.com/cell/fulltext/S0092-8674(18)30154-5

<h3> Type of Machine Learning Problem</h3>

<p> It is a one class classification problem, for a given image we need to predict if they are suffering from which disease. </p>


<h3> How does it work ? </h3>
* Understanding image data <br>
* Image augmentation <br>
* Applying model <br>

<h3> Conclusion </h3>
1. we applied image augmentation to this dataset.<br>
2. we applied the model - MobileNetV2. <br>
3. we have taken the weight of every model which is trained on imagenet dataset, we have not freezed the layer because the retina dataset is different from imagenet dataset.<br>
4. we used confusion matrix because dataset is imbalanced and so accuracy score may not give good sence of result.<br>
5. By seeing the confusion matrix of all the three model we can say that inception net has best recall  than other model, precision is also good for inception net.
