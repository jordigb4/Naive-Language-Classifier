# Naive-Language-Classifier

<a name="readme-top"></a>

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/albertojc6/Naive-Language-Classifier">
    <img src="images/logo.png" alt="Logo" width="200" height="130">
  </a>

<h3 align="center">European Language Identifier</h3>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#repo-structures">Repository Structures</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## About The Project

This is a project to develop a language identifier for the following European languages: English, Spanish, Dutch, German, Italian, and French. In this regard, the 'wortschatz leipzig corpora' has been utilized, consisting of 30k and 10k sentences for each language, for the training set and the test set, respectively. 
 
Its primary goal is academic, particulary introduce NLP concepts and tools while analyzing which text processing techniques are most effective for distinguishing languages. However, the project also has practical purposes, as language identification has wide-ranging applications in NLP. It serves as a foundational step in text classification tasks, aiding in the categorization of text based on language for subsequent analysis or processing. The task of language id is thus the first step in most language processing pipelines.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running follow these simple example steps.

### Prerequisites


* Python
  ```sh
  pip install nltk
  pip install numpy
  ```

### Installation

1. Clone the repo
  ```sh
  git clone https://github.com/jordigb4/Naive-Language-Classifier.git
  ```
<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- USAGE EXAMPLES -->
## Usage

```python
from language_identifier import LanguageIdentifier

# Train classifier
LangId = LanguageIdentifier()

#Predict phrase or
pred = LangId.identify_language('The classifier must be used like this', smoothing = 'Lidstone')
#Predict valid text file
pred = LangId.identify_language(f"corpora/eng_tst.txt", smoothing = 'Lidstone')

print(pred)
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Repository Structures

    .
    ├── corpora                 # Train, val, and test corpora files (.txt)
    ├── images                  # Resulting Confusion Matrices
    ├── Preprocessing           # File with functions to treat raw corpora
    ├── language_identifier     # MAIN file, with classifier model class
    ├── test.ipynb              # Test language_identifier accuracy, both Lidstone and Abs. Discounting 
    ├── wrong_classified.ipynb  # Obtain wrong classified phrases from test corpora
    ├── all_wrong_*.csv         # Compilation of misclassified phrases
    ├── foreign_languages.ipynb # Experiment to check potential model BIAS
    ├── best_alpha.py           # Obtain best Lidstone parameter for data in corpora dir.
    └── README.md

<p align="right">(<a href="#repo-structures">back to top</a>)</p>

<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTACT -->
## Contact

Your Name - [@your_twitter](https://twitter.com/your_username) - email@example.com

Project Link: [https://github.com/jordigb4/Naive-Language-Classifier](https://github.com/jordigb4/Naive-Language-Classifier)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

* [Naive Bayes Classifier Paper](https://web.stanford.edu/~jurafsky/slp3/)

<p align="right">(<a href="#readme-top">back to top</a>)</p>