### How to Implement the Travel Style Prediction Model

To use the trained model in your external Python code or within this notebook, follow these steps:

#### 1. Requirements
Ensure you have the following libraries installed:
```bash
pip install pandas joblib scikit-learn
```

#### 2. Loading the Model
The model is saved as a scikit-learn Pipeline within a `.pkl` file. You can load it using `joblib`:

```python
import joblib
import pandas as pd

# Path to your exported file
model_path = 'random_forest_travel_model.pkl'
model = joblib.load(model_path)
```

#### 3. Expected Features
The model expects a pandas DataFrame with exactly 8 numerical features. Values should typically range from 0.0 to 1.0:

| Feature | Description |
| :--- | :--- |
| `active_movement` | Preference for physical activities/hiking |
| `relaxation` | Preference for downtime and spas |
| `cultural_interest` | Interest in museums, history, and local art |
| `cost_sensitivity` | Importance of staying within a low budget |
| `luxury_preference` | Preference for high-end services and comfort |
| `family_friendliness` | Suitability for children and family groups |
| `nature_orientation` | Focus on outdoor and natural environments |
| `social_group` | Preference for group activities and nightlife |

#### 4. Making Predictions
Input data must be a DataFrame with column names matching the list above:

```python
data = pd.DataFrame([{
    'active_movement': 0.8, 
    'relaxation': 0.2, 
    'cultural_interest': 0.1, 
    'cost_sensitivity': 0.1, 
    'luxury_preference': 0.9, 
    'family_friendliness': 0.0, 
    'nature_orientation': 0.9, 
    'social_group': 0.1
}])

# Get the class label (e.g., 'Adventure')
prediction = model.predict(data)[0]

# Get the confidence scores for each category
probabilities = model.predict_proba(data)
```