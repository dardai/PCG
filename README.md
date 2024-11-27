## SCL-BT

Implementation of SCL-BT, a Structural Contrastive Learning-based Bug Triaging framework.


### Main File Contents

- `/bug_Augmentation/preprocess.py` includes text cleaning, tokenization and parts for bug report preprocessing.
- `/bug_Augmentation/augment_main.py` implement prototype clustering-based augmentation.
- `SCL-BT.py` is the joint model of GCF backbone and SSL module
- `main.py` is the training and validating progresses for SCL-BT.

### How to use

1. Install required packages (Using a virtual environment is recommended).
   `pip install -r requirements.txt`
2. Download datasets into the repository as following:
    - Download `gc.json`, `classifier_data_0.json`, `classifier_data_5.json`, `classifier_data_10.json`
      , `classifier_data_20.json` from [here](https://pan.baidu.com/s/1LMOUD8_fWNgMPQLawkhhXA?pwd=5zcn), then put them
      into **/bug_Augmentation/data/google_chromium** folder.
    - Download `mc.json`, `classifier_data_0.json`, `classifier_data_5.json`, `classifier_data_10.json`
      , `classifier_data_20.json` from [here](https://pan.baidu.com/s/1jyac2q5Ak7GJqcJf9t9Gvw?pwd=tg9w), then put them
      into **/bug_Augmentation/data/mozilla_core** folder.
    - Download `mf.json`, `classifier_data_0.json`, `classifier_data_5.json`, `classifier_data_10.json`
      , `classifier_data_20.json` from [here](https://pan.baidu.com/s/1-Q33mN2SUAQhygr30Xeyrw?pwd=c9o2), then put them
      into **/bug_Augmentation/data/mozilla_firefox** folder.
3. Run `preprocess.py`.
   ```python
   cd bug_Augmentation
   python preprocess.py
   ```
4. Run `augment_main.py`.
   ```python
   cd bug_Augmentation
   python augment_main.py
   ```
5. Run `main.py`.
   ```python
   python main.py
   ```

### Contribution

Any contribution (pull request etc.) is welcome.

### Data Sample

A sample bug report from datasets is given below:

#### Google Chromium:

```json
{
		"id" : 1,
		"issue_id" : 2,
		"issue_title" : "Testing if chromium id works",
		"reported_time" : "2008-08-30 16:00:21",
		"owner" : "jack@chromium.com",
		"description" : "\nWhat steps will reproduce the problem?\n1.\n2.\n3.\n\r\nWhat is the expected output? What do you see instead?\n\r\n\r\nPlease use labels and text to provide additional information.\n \n ",
		"status" : "Invalid",
		"type" : "Bug"
}
```
