# Installation Guide

Follow these steps to set up your environment and install the necessary packages for this project.

---

## **1. Create a Virtual Environment**
To keep dependencies isolated, create a Python virtual environment:
```bash
python3 -m venv <env_name>
```

Activate the environment:
- On **Linux/Mac**:
  ```bash
  source <env_name>/bin/activate
  ```
- On **Windows**:
  ```bash
  <env_name>\Scripts\activate
  ```

---

## **2. Upgrade `pip`**
Ensure you have the latest version of `pip`:
```bash
pip install --upgrade pip
```

---

## **3. Install Required Packages**
### **3.1. Install NeMo Toolkit**
Install a specific version of the NVIDIA NeMo toolkit:
<!-- python -m pip install git+https://github.com/NVIDIA/NeMo.git@52d50e9e09a3e636d60535fd9882f3b3f32f92ad -->
```bash
python -m pip install git+https://github.com/NVIDIA/NeMo.git
```

### **3.2. Install WhisperX**
Install WhisperX using its GitHub repository:
```bash
python -m pip install git+https://github.com/m-bain/whisperx.git
```

---

## **4. Install `youtokentome`**
To install `youtokentome`, follow these steps:
1. Install **Cython**:
    ```bash
    pip install Cython
    ```
2. Install **youtokentome**:
    ```bash
    python -m pip install git+https://github.com/gburlet/YouTokenToMe.git@dependencies
    ```

---

## **5. Install Remaining Dependencies**
Install all other dependencies from the `requirements.txt` file:
```bash
pip install -r requirements.txt
```

## **6. Install ffmeg to you device**
```bash
sudo apt update 
sudp apt install ffmpeg
```

---

## **Common Issues**
### **Error:**
```text
ImportError: cannot import name 'ModelFilter' from 'huggingface_hub'
```
This error may occur when importing:
```python
from nemo.collections.asr.models import EncDecMultiTaskModel
```

### **Solution:**
The issue is due to a version conflict in the `huggingface-hub` package. To resolve this, install `huggingface-hub==0.23.2`