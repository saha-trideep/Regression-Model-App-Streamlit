# Import Libraries
import os
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# Read & Clean the data
#-----------------------------------------------------#
def get_data():
    data = pd.read_csv("data/data.csv") #read the data
    
    # we are going to drop "id" & "Unnamed: 32" columns
    data = data.drop(['Unnamed: 32', 'id'], axis=1)

    # Converting existing column "diagnosis" into binary
    data['diagnosis'] = data['diagnosis'].apply(lambda x: 1 if x=="M" else 0)
    
    return data

#------------------------------------------------------#
# Model Building Pipeline
#------------------------------------------------------#

class MLPipeline:
    def __init__(self, data):
        self.data = data
        self.x_test, self.x_train, self.y_test, self.y_train = None, None, None, None
        self.model = None
        self.scaler = None
    
    def preprocessing_data(self):
        # divide target & predictor variables
        y = self.data['diagnosis']
        x = self.data.drop(['diagnosis'], axis=1)
        
        # normalize
        self.scaler = StandardScaler()
        x_scaled = self.scaler.fit_transform(x)
        
        # train | test 
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42)
        
    
    def train_model(self):
        self.model = LogisticRegression()
        self.model.fit(self.x_train, self.y_train)
    
    # test | save 
    def test_model(self):
        y_pred = self.model.predict(self.x_test)
        accuarcy = accuracy_score(self.y_test, y_pred)
        clssifc_report = classification_report(self.y_test, y_pred)
        return f"Accuracy: {accuarcy: .2f}\nClassification Report:\n{clssifc_report}"

    def save_model(self, folder_path = "models", model_filename="trained_model.joblib"):
        os.makedirs(folder_path, exist_ok=True)
        model_path = os.path.join(folder_path, model_filename)
        joblib.dump(self.model, model_path)
    
    def save_scaler(self, folder_path = "scalers", scaler_filename="scaler.joblib"):
        os.makedirs(folder_path, exist_ok=True)
        scaler_path = os.path.join(folder_path, scaler_filename)
        joblib.dump(self.scaler, scaler_path)
    

def main():
    #print("Hello World")
    
    data = get_data()
    #print(data.head())
    pipeline = MLPipeline(data=data)
    pipeline.preprocessing_data()
    pipeline.train_model()                      
    #print(pipeline.test_model())               # Accuracy:  0.97
                                                # Classification Report:
                                                #            precision    recall  f1-score   support
                                                #
                                                #        0       0.97      0.99      0.98        71
                                                #        1       0.98      0.95      0.96        43
                                                #
                                                #    accuracy                         0.97       114
                                                #    macro avg     0.97      0.97     0.97       114
                                                #  weighted avg    0.97      0.97     0.97       114
    pipeline.save_model()
    pipeline.save_scaler()
                                                
                                                
                                                
     
if __name__=='__main__':
    main()