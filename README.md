![Python](https://img.shields.io/badge/-Python-05122A?style=flat&logo=python)&nbsp;

## *Chara*
<br /> <br />


# Navigation
 - [Setting up a project](#setting_up_a_project)
 - [Project structure](#project_structure)
<br /> <br />


<a name="setting_up_a_project"></a> 
## Setting up a project
 - Select the Python version: 3.11.0rc1 WSL
 - Launch Virtual Environment in command line Ubuntu WSL in Pycharm: `source /home/<username>/.virtualenvs/Chara/bin/activate`
 - Install dependencies:  `pip install -r requirements.txt`
<br /> <br />


<a name="project_structure"></a> 
# Project structure
    Chara
    ├── module/
    │   ├── __init__.py
    │   ├── convert_image_name.py
    │   ├── create_model.py
    │   ├── edit_image_name.py
    │   └── load_train_test_val.py
    ├── images/
    │   └── __init__.py
    ├── tests/
    │   ├── __init__.py
    │   ├── test_convert_image_name.py
    │   ├── test_create_model.py
    │   ├── test_edit_image_name.py
    │   ├── test_load_train_test_val.py
    ...
    │
    ├── DockerFile                      # Launch Project in Docker
    ├── README.md                       # Project documentation
    └── requirements.txt                # File with dependencies
<br /> <br />
