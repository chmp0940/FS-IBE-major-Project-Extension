# Instructions for the Meta-IDS Team

## Your Independent Module and Future Handoff


Your responsibility is to develop an independent **network-level Meta-IDS module**. The module must analyze IoMT/IoT network traffic and return a clear, documented network-risk result. This result will be connected later to a separate forward-secure encrypted-query system.

You should complete, test, and document your module independently first. You do not need to work inside the cryptographic implementation while developing this module.


---

## 1. Project Idea in Simple Words

There are two security problems:

1. A device may send suspicious network traffic.
2. A user may send suspicious queries even when the user's signature is valid.

Your module will solve the network-layer problem. The existing encryption project will handle the query and access-control problem:

| Layer | Responsible team | Main question |
|---|---|---|
| Network layer | Meta-IDS team | Is the network traffic normal or suspicious? |
| Query and access-control layer | Forward-Secure IBE team | Should this user/query be allowed to access encrypted IoT data? |

The combined system will be:

```text
Network traffic
        |
        v
Meta-IDS network-risk score
        |
        +--------------------+
        |                    |
        v                    v
Query behavior          Signature and epoch
ML trust score          verification
        |                    |
        +---------+----------+
                  |
                  v
       Final adaptive trust decision
                  |
        +---------+---------+
        |                   |
        v                   v
Allow encrypted query   Reject or restrict query
```

The final research direction is:

> Use network-level intrusion detection together with query-level behavioral trust to control access to forward-secure encrypted IoT data.

This is more useful than simply comparing two ML algorithms because the ML result will control a real security action.

---

## 2. Your Development Boundary

You must implement and test your Meta-IDS module independently.


The existing Forward-Secure IBE project will connect your completed module later. Your work should focus only on:

- Network-flow data
- Data preprocessing
- Feature selection
- Base learners
- Meta-learning
- Anomaly detection
- Network-risk output
- Network-level evaluation

Do not directly import or edit the cryptographic modules during the independent development phase.

---

## 3. Current Project Structure

### 3.1 Existing Forward-Secure IBE files

| File | Responsibility |
|---|---|
| `lattice_infrastructure.py` | Lattice parameters, gadget matrix, trapdoor-related prototype, identity hashing, binary epoch tree |
| `forward_security.py` | Epoch-related key handling, encryption, decryption, minimal-cover-style update logic |
| `Trust_Model.py` | Query structure, signature simulation, trust score, query validation, query feature extraction |
| `simulation.py` | Full encryption, query, trust, matching, decryption, and benchmark workflow |
| `attack_simulator.py` | Synthetic normal and attack behavior generation for the query ML model |
| `ml_trust_engine.py` | Isolation Forest and Random Forest query-behavior model |
| `ml_simulation.py` | Static trust versus ML-enhanced trust benchmark |
| `Major_Project_ML_Trust_Scoring.ipynb` | Notebook explaining and demonstrating the query-level ML trust system |

### 3.2 Your current Meta-IDS files

| File | Current purpose |
|---|---|
| `Meta_IDS_Simple_Preprocessing_Base_Learners.ipynb` | Dataset preprocessing and base-learner experiments |
| `Meta-IDS_Meta-Learning-Based_Smart_Intrusion_Detection_System_for_Internet_Of_Medical_Things_IoMT_Network.pdf` | Reference paper describing the complete Meta-IDS approach |

Your current notebook is a useful starting point, but it is not yet a complete implementation of every part of the paper.

---

## 4. What the Reference Meta-IDS Paper Proposes

The reference paper is:

> *Meta-IDS: Meta-Learning-Based Smart Intrusion Detection System for Internet of Medical Things (IoMT) Network*

The paper has two main detection paths.

### 4.1 Known-attack detection

The paper describes several base learners:

- Decision Tree
- Random Forest
- AdaBoost
- Multinomial Naive Bayes
- Neural Network
- Multilayer Perceptron

Their predictions are combined using a meta-learner, described in the paper as XGBoost.

The idea is:

```text
Network features
        |
        v
Several base learners
        |
        v
Meta-learner
        |
        v
Known-attack prediction
```

### 4.2 Unknown or anomaly detection

The paper also describes a separate anomaly-detection path:

```text
Suspicious or remaining traffic
        |
        v
Mean-shift clustering
        |
        v
Normal or suspicious cluster
        |
        v
Biased classifiers for uncertain samples
        |
        v
Unknown-attack decision
```

The purpose is to detect behavior that was not clearly represented in the supervised training labels.

### 4.3 Paper datasets

The paper evaluates its method on:

- WUSTL-EHMS-2020
- IoTID20
- WUSTL-IIOT-2021

The current notebook uses the WUSTL-EHMS-2020 dataset:

```text
wustl-ehms-2020_with_attacks_categories.csv
```

The other two datasets are useful future validation datasets, but they are not required for the first independent module.

---

## 5. What the Meta-IDS Team Has Already Implemented

The current notebook already includes the following:

### 5.1 Dataset loading

It loads the WUSTL-EHMS-2020 CSV file and displays:

- Dataset shape
- Example rows
- Missing values
- Label distribution

### 5.2 Column cleanup

The notebook removes selected columns such as:

- `Dir`
- `SrcAddr`
- `DstAddr`
- `Sport`
- `SrcMac`
- `DstMac`
- `Attack Category`

The target column is:

```text
Label
```

### 5.3 Train/test split

The notebook uses a stratified 70%/30% split:

```text
70% training data
30% test data
```

The test data is kept separate from training.

### 5.4 Preprocessing

The notebook currently performs:

- Numeric missing-value filling using the training median
- Categorical missing-value filling using the training mode
- One-hot encoding
- Train/test feature-column alignment
- StandardScaler normalization

### 5.5 Class balancing

SMOTE is applied to the training data only.

This is correct because the test set should represent the original evaluation distribution.

### 5.6 Feature selection

Random Forest Recursive Feature Elimination selects up to 15 features.

### 5.7 Base learners

The notebook currently trains and evaluates:

- Decision Tree
- Random Forest
- AdaBoost
- KNN
- MLP
- Multinomial Naive Bayes

The notebook calculates:

- Accuracy
- Precision
- Recall
- F1-score

---

## 6. What Is Not Yet Implemented in the Current Notebook

The current notebook does not yet represent the complete paper implementation.

The following parts still need to be considered:

### Required for the first Meta-IDS module

1. A clear base-learner output format.
2. A real meta-learner that combines base-learner predictions.
3. A complete saved preprocessing pipeline.
4. A saved model artifact that can be loaded later.
5. A single network-risk output interface.
6. Per-attack evaluation.
7. A confusion matrix.
8. A reproducible results file.

### Optional extensions

These can be added after the first module works:

- Mean-shift clustering
- Unknown-attack detection
- Biased classifiers for uncertain samples
- XGBoost meta-learner
- Hyperparameter optimization
- Multiple datasets
- Cross-validation
- Time-based or device-based holdout testing

Do not attempt all optional extensions at the same time.

---

## 7. Exact Work You Should Implement

Complete your work in the following stages.

## Stage 1: Make the current base-learner experiment reproducible

The team should:

1. Load the WUSTL-EHMS-2020 dataset.
2. Apply preprocessing using training data only.
3. Apply SMOTE only to the training split.
4. Apply feature selection only using training data.
5. Train the six base learners.
6. Save the selected feature names.
7. Save the preprocessing objects.
8. Save the trained base learners.
9. Generate one results table.
10. Generate confusion matrices.
11. Save the final results to CSV or JSON.

The output should be reproducible using a fixed random seed.

## Stage 2: Add a meta-learner

The meta-learner should receive the outputs of the base learners.

For every input sample, the base learners produce predictions:

```text
Decision Tree prediction
Random Forest prediction
AdaBoost prediction
KNN prediction
MLP prediction
Naive Bayes prediction
```

These predictions become the input to the meta-learner:

```text
[DT_output, RF_output, AdaBoost_output, KNN_output, MLP_output, MNB_output]
```

The meta-learner produces:

```text
normal or attack
```

Preferably, it should also produce:

```text
attack probability
```

The team must clearly document whether it uses:

- Hard predictions: `0` or `1`
- Probability predictions: for example, `0.0` to `1.0`
- Both hard predictions and probabilities

Probability outputs are generally more useful for the later risk-fusion stage.

## Stage 3: Add a network-risk interface

The independent Meta-IDS module should provide a function with a stable interface.

The exact implementation language can be Python, but the interface should conceptually be:

```python
result = evaluate_network_window(network_features)
```

The result should contain:

```python
{
    "network_risk": 0.0,
    "is_network_anomaly": False,
    "attack_type": "normal",
    "confidence": 0.0
}
```

The values have the following meaning:

| Field | Meaning |
|---|---|
| `network_risk` | A value from 0.0 to 1.0; higher means more suspicious |
| `is_network_anomaly` | Boolean network-level decision |
| `attack_type` | Predicted attack category or `"unknown"` |
| `confidence` | Confidence from 0.0 to 1.0 |

The later integration should not depend on notebook display output. It should use this structured result.

## Stage 4: Save and load the complete pipeline

You should save:

- Preprocessing/scaler object
- Encoding column names
- Selected feature names
- Feature-selection object
- Base learners
- Meta-learner
- Label mapping
- Model version

The final module should be loadable without retraining.

Example concept:

```python
network_detector = load_network_detector("meta_ids_model")
result = network_detector.evaluate(network_features)
```

The existing Forward-Secure IBE project will later connect to this interface.

---

## 8. Optional Unknown-Attack Module

After the supervised Meta-IDS module works, you can implement the anomaly path.

The anomaly module should:

1. Receive network feature vectors.
2. Detect clusters using mean-shift or another documented anomaly method.
3. Assign clusters as normal or suspicious.
4. Produce an anomaly probability or risk score.
5. Identify uncertain samples.
6. Use a secondary classifier for uncertain samples if required.

The output should use the same interface:

```python
{
    "network_risk": 0.0,
    "is_network_anomaly": False,
    "attack_type": "unknown",
    "confidence": 0.0
}
```

The team must report separately:

- Known-attack detection results
- Unknown/anomaly detection results

Do not combine the two result tables without explaining the difference.

---

## 9. What You Should Not Implement

You should not implement or modify:

- Lattice trapdoor generation
- Lattice encryption
- Lattice decryption
- Identity-based key generation
- Epoch key evolution
- Query signing
- Signature verification
- Trust-score storage
- Encrypted-data matching
- Forward-secure key deletion

Those belong to the existing Forward-Secure IBE project. Your module only needs to return a reliable network-risk result.

---

## 10. Interface Between the Two Teams

The teams should agree on the following interface before merging.

### Input to Meta-IDS

The input is one network-flow window or one network-behavior record.

It should contain the same features used during training, in the same order and format.

The input must not contain:

- The true label
- The final access decision
- The user's trust score
- Future information from the test period

### Output from Meta-IDS

The minimum output should be:

```python
{
    "network_risk": float,
    "is_network_anomaly": bool,
    "confidence": float
}
```

The preferred output also contains:

```python
"attack_type": str
```

### Important rule

The later integration will treat your Meta-IDS output as evidence, not as an absolute cryptographic decision.

For example:

- Invalid signature should still be rejected even if network risk is low.
- A valid signature should not automatically be accepted if network and query behavior are highly suspicious.
- A high network risk should influence trust but should not silently bypass epoch and signature checks.

---

## 11. How the Final Combined System Will Work

The final system will have two ML risk sources.

### 11.1 Network risk

Produced by the Meta-IDS team:

```text
R_network
```

### 11.2 Query risk

Produced by the Forward-Secure IBE team's ML trust engine:

```text
R_query
```

The query model currently uses features such as:

- Query frequency
- Epoch mismatch rate
- Signature failure rate
- Inter-query interval
- Payload-size deviation
- Time-of-day encoding
- Query burst count

### 11.3 Security facts

Some conditions are not ordinary ML evidence:

- Invalid signature
- Invalid query serialization
- Unauthorized identity
- Invalid or expired epoch

These conditions must be handled by the security validator.

### 11.4 Future risk-fusion layer

The final integration may combine:

```text
network risk
query risk
signature result
epoch result
```

Conceptually:

```text
final risk =
    network contribution
    + query contribution
    + signature/epoch security penalties
```

The exact formula will be selected and documented during integration. You do not need to decide the final fusion formula.

---

## 12. Suggested Adaptive Actions

The final system may use risk ranges:

| Risk range | Possible action |
|---:|---|
| 0.00–0.25 | Allow query |
| 0.25–0.50 | Allow and monitor |
| 0.50–0.70 | Rate-limit or request re-authentication |
| 0.70–0.90 | Reject query and reduce trust |
| 0.90–1.00 | Block or quarantine device |

These actions are part of the final integration, not the independent Meta-IDS module.

You only need to produce a reliable network-risk value.

---

## 13. Data Integration Rules

The two datasets must not be joined randomly.

One network-flow row should not be randomly paired with an unrelated query row.

The correct future integration unit is a session or time window:

```text
One device/session/time window
    |
    +-- network features
    +-- query behavior features
    +-- signature results
    +-- epoch behavior
    +-- attack label
```

The combined dataset should ideally contain:

| Field | Description |
|---|---|
| `device_id` | Device or session identifier |
| `time_window` | Time range of the observation |
| `network_features` | Network-flow features |
| `query_features` | Query behavior features |
| `signature_status` | Valid or invalid signature information |
| `epoch_status` | Correct or incorrect epoch information |
| `attack_type` | Attack category |
| `label` | Normal or malicious |

If a real shared session identifier is not available, the teams must document how the experimental episodes were created.

---

## 14. Recommended Evaluation Plan

The final paper should compare the following systems.

### Baseline A: Cryptographic static trust

```text
Forward-secure IBE
+ signature verification
+ static trust score
```

### Baseline B: Query-level ML trust

```text
Forward-secure IBE
+ signature verification
+ query behavior ML
```

### Baseline C: Network Meta-IDS only

```text
Network features
+ Meta-IDS
```

### Proposed System D: Combined cross-layer system

```text
Network Meta-IDS
+ query behavior ML
+ signature verification
+ epoch validation
+ forward-secure encrypted access control
```

The paper should report:

- Accuracy
- Precision
- Recall
- F1-score
- False-positive rate
- False-negative rate
- False Trust Acceptance Rate
- False Trust Rejection Rate
- Per-attack detection
- Query latency
- Throughput
- ML inference time
- Encryption time
- Decryption time
- Memory usage where possible

---

## 15. Required Attack Scenarios

The joint evaluation should include:

### Network-level scenarios

- Flood or denial-of-service behavior
- Spoofing
- Man-in-the-middle-like modification
- Injection
- Port scanning or reconnaissance
- Backdoor-like behavior where available in the dataset

### Query-level scenarios

- Invalid signature
- Valid-signature flood
- Replay query
- Wrong or expired epoch
- Unusual query burst
- Payload probing
- Repeated suspicious behavior by one identity

### Combined scenarios

The most useful scenarios are:

1. Normal network and normal query.
2. Malicious network and normal query.
3. Normal network and malicious query.
4. Malicious network and malicious query.
5. Valid signature but malicious network behavior.
6. Valid signature but malicious query behavior.
7. Previously unseen or synthetic unknown behavior.

This will show why both layers are necessary.

---

## 16. How to Start Work on a New Module

Before writing code, create a short module plan containing:

1. Module name.
2. Purpose.
3. Input format.
4. Output format.
5. Files that will be created.
6. Files that must not be changed.
7. Dataset used.
8. Evaluation metrics.
9. How the module will be tested.

For the Meta-IDS team, the first module plan should be:

```text
Module: NetworkRiskDetector

Purpose:
Return a network anomaly score from a network-flow window.

Input:
Preprocessed network features using the saved training schema.

Output:
network_risk, is_network_anomaly, confidence, attack_type.

Must not modify:
All Forward-Secure IBE and query-trust files.

Validation:
Run the detector on a fixed test set and save metrics.
```

---

## 17. How to Resume an Existing Module

When resuming work:

1. Read this guide first.
2. Read the module's README or notebook.
3. Check the current Git status.
4. Identify which cells or files are already complete.
5. Run the existing test or evaluation before editing.
6. Write down the current result.
7. Continue only from the first incomplete step.
8. Do not silently change the dataset split or feature order.
9. Do not replace a model without recording the reason.
10. Update the module's result file after the change.

A resumed module should always record:

```text
Date:
Module:
Current status:
Last completed step:
Next step:
Dataset:
Random seed:
Current metrics:
Known limitations:
```

This can be kept in the module's README or experiment log.

---

## 18. Your Required Deliverables

You should deliver the following independently:

### Documentation

- A short README explaining how to run the module.
- Feature-list documentation.
- Dataset and label documentation.
- Model architecture description.
- Known limitations.

### Code

- Preprocessing module.
- Base-learner training module.
- Meta-learner module, when completed.
- Network-risk inference module.
- Model-save and model-load functions.

### Artifacts

- Saved preprocessing object.
- Saved feature-selection object.
- Saved base learners.
- Saved meta-learner.
- Feature schema file.
- Results CSV or JSON.
- Confusion matrix.
- Classification report.

### Test

A minimal test must demonstrate:

```text
Input network record
        |
        v
Loaded model
        |
        v
Structured network-risk output
```

The final output should not require a notebook display cell.

---

## 19. Definition of Done

The Meta-IDS module is complete when:

- The notebook or Python module runs from a clean environment.
- The dataset path is documented.
- The random seed is fixed.
- The train/test split is reproducible.
- Test data is not used to fit preprocessing.
- SMOTE is applied only to training data.
- Feature order is saved.
- The model can be saved and loaded.
- One network-risk function works on new input.
- The output format is documented.
- Metrics are saved.
- Per-attack results are reported where labels are available.
- Known limitations are written down.
- No cryptographic project files were modified.

---

## 20. Important Technical Cautions

### Do not report only accuracy

Intrusion datasets can be imbalanced. Always report:

- Precision
- Recall
- F1-score
- Confusion matrix
- False-positive rate
- False-negative rate

### Avoid data leakage

Do not fit these using the test set:

- Scaler
- Encoder
- SMOTE
- Feature selector
- Model

### Keep preprocessing with the model

A trained model is not enough. The same feature order and preprocessing steps must be available during inference.

### Do not claim zero-day detection from an ordinary random split

If the same attack pattern exists in both training and testing data, the model may only be recognizing a known pattern.

For stronger zero-day evaluation, hold out:

- An attack category
- A device
- A time period
- A network environment

### Be careful with high accuracy

Very high accuracy does not automatically prove real-world security. Check:

- Per-attack recall
- False negatives
- Cross-dataset performance
- Time-based testing
- Resource usage
- Unseen attack behavior

---

## 21. Prototype Limitations of the Existing Combined Project

The Forward-Secure IBE project is a research prototype.

The current code documents that:

- The lattice trapdoor and preimage-sampling implementation are pedagogical.
- The current signature class is an HMAC-based simulation placeholder, not a production Dilithium-3 implementation.
- The query ML dataset includes synthetic attack behavior.
- The Meta-IDS notebook currently contains preprocessing and base learners, not the entire paper pipeline.

The final paper must state these limitations honestly.

The project should not claim:

- Production-grade Dilithium-3 security before a real implementation is integrated.
- A complete reproduction of the Meta-IDS paper before its missing stages are implemented.
- Real-world zero-day protection based only on synthetic or random-split evaluation.

---

## 22. Your Main Responsibility

Focus on this sentence:

> “We will build an independent network-risk detector that converts IoMT network-flow behavior into a reliable, documented risk score.”

After you deliver this module, it will be connected to the existing system as follows:

> “Your network-risk output will be combined with query behavior, signature verification, and epoch validation to control access to forward-secure encrypted IoT data.”

The combined research contribution will be:

> “A cross-layer adaptive IoT security system that connects network-level Meta-IDS detection with query-level ML trust and forward-secure cryptographic access control.”

Integration should happen only after you deliver:

1. A stable input feature schema.
2. A saved preprocessing pipeline.
3. A saved trained model.
4. A documented inference function.
5. Structured network-risk output.
6. Reproducible evaluation results.
