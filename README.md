# Language-Classification-using-SVM
Using machine learning techniques to classify different languages.

Abstract

Language classification is a fundamental task in natural language processing (NLP), with applications ranging from text mining to machine translation. In this report, we explore the use of machine learning techniques, specifically SVM Algorithm, for language classification using Python. We preprocess the data, engineer relevant features, and analyze their correlations to build an effective model. We also employ Principal Component Analysis (PCA) to reduce feature dimensionality and improve model performance. Our results demonstrate the effectiveness of the approach in accurately classifying languages.

About Dataset

The dataset that we have used consists of text data of 3 languages, English, Afrikaans, Nederlands. We have almost 3000 rows of such data, each labelled with their corresponding labels. We have manipulated this data to create more features. Our analysis began with data preprocessing, where we addressed missing values and converted the text and language columns to string type for further processing. We then proceeded to engineer a diverse set of features from the text data, including word count, character count, punctuation count, vowel count, capitalization features, and more. These features were carefully designed to capture linguistic characteristics and patterns that differentiate languages.
Exploratory data analysis revealed insights into the distribution of features and their correlations, providing valuable context for model development. Visualization techniques such as heatmaps helped us identify relationships between variables and assess their impact on language classification.
Making extra features from existing dataset is one way to improve model’s performance. The impact may vary from model to model but for us it worked really well. Further PCA has been done to obtain 12 components which have 95% variance of the original features. This enables us to do a better classification task. When working with many columns of data it is beneficial to use PCA or other algorithms to reduce the dimensions and keep limited variance in the dataset at cost of reducing complexity. Train test split by sklearn has been used to split the data into 80-20 set for training and testing. There are around 2200 rows in training set and around 550 in test set.

![image](https://github.com/kar5960/Language-Classification-using-SVM/assets/156512487/90a19d2c-6823-48fc-8617-ecc9ad49c186)

Training on SVM

With the feature set defined, we explored a range of machine learning algorithms for language classification, including Support Vector Machines (SVM). We leveraged Python's scikit-learn library to train SVM classifiers and evaluate their performance using accuracy as the primary metric. SVMs, with a linear kernel, were chosen for their ability to handle high-dimensional data and their effectiveness in separating classes in feature space.

The model achieved a commendable accuracy on the test set, indicating its ability to generalize well to unseen data. Additionally, the confusion matrix provided insights into the model's performance across different language classes, highlighting areas of strength and potential improvement.

Results

The accuracy on test set comes out to be 89%. This is quite good. We can see from the confusion matrix below that the model has struggled to learn Nederlands as the predictions are quite poor. One of the main reasons for this is not enough data. Very few datapoints were available for testing and training of this label. Imbalanced datasets do produce such issues. Creating more data synthetically was an option that we did not explored for making more data points such that the dataset can be balanced. But that can be explored.  
![image](https://github.com/kar5960/Language-Classification-using-SVM/assets/156512487/0c41457d-c1b6-426c-b24a-eaaa4bf4a055)

Conclusion and Future Directions

In conclusion, our exploration of language classification using machine learning techniques has provided valuable insights into the complexities of text analysis and model development. By leveraging Python's rich ecosystem of libraries and tools, we have demonstrated the feasibility of building accurate and efficient language classification models.

While the current model has shown promising results, there are several avenues for future research and improvement. Experimentation with alternative algorithms, such as deep learning architectures like recurrent neural networks (RNNs) or transformers, could offer further improvements in accuracy and generalization. Additionally, fine-tuning hyperparameters and exploring ensemble methods may enhance the model's robustness and stability.

