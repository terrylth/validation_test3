# Internal Audit Finding Escalation Classifier — Model Validation Exercise

## Overview

This repository contains a binary classification model that predicts whether
an internal audit finding will be escalated to the Board Risk Committee.

You are acting as an independent model validator. Your task is to:

1. Run the model script and observe the outputs carefully
2. Review the model script for any issues
3. Review the test script and assess whether validation checks are correctly implemented
4. Document your findings

---

## Repository Structure

```
validation_test/
├── model_script_v4.py    # Model training script
├── test_script_v4.py     # Validation and testing suite
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

---

## Setup Instructions

**Step 1 — Clone the repository**
```bash
git clone <repo-url>
cd validation_test
```

**Step 2 — Create and activate a virtual environment**
```bash
python3 -m venv interview_env
source interview_env/bin/activate
```

**Step 3 — Install dependencies**
```bash
pip install -r requirements.txt
```

---

## Running the Scripts

Run the model training script first:
```bash
python3 model_script_v4.py
```

Then run the validation suite:
```bash
python3 test_script_v4.py
```

---

## Notes

- Data is synthetically generated — no external data file required
- The model uses both structured features and text features (finding descriptions)
- Target variable: `escalated` (1 = escalated to Board Risk Committee, 0 = resolved operationally)
