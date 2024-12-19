from db_conn import *
from pprint import pprint
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from scipy.special import expit
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

class class_loan_classification():
    def __init__(self, import_data_flag=True):
        self.conn, self.cur = open_db()
        if import_data_flag:
            self.import_loan_data()
            
    def import_loan_data(self):
        drop_sql =""" drop table if exists loan;"""
        self.cur.execute(drop_sql)
        self.conn.commit()
    
        create_sql = """
            create table loan (
                Loan_ID varchar(255) PRIMARY KEY,
                Gender varchar(255),
                Married varchar(255),
                Dependents varchar(255),
                Education varchar(255),
                Self_Employed varchar(255),
                ApplicantIncome float,
                CoapplicantIncome float,
                LoanAmount float,
                Loan_Amount_Term float,
                Credit_History INT,
                Property_Area varchar(255),
                Loan_Status varchar(255)
                ); 
        """
    
        self.cur.execute(create_sql)
        self.conn.commit()
    
        file_name = './data/loan.data.csv'
        loan_data = pd.read_csv(file_name)

        for col in loan_data.columns:
            loan_data[col].fillna(0, inplace=True)
        
        rows = []
    
        insert_sql = """insert into loan (Loan_ID,Gender,Married,Dependents,Education,Self_Employed,ApplicantIncome,CoapplicantIncome,LoanAmount,Loan_Amount_Term,Credit_History,Property_Area,Loan_Status)
                        values(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s);"""

        for t in loan_data.values:
            rows.append(tuple(t))
    
        self.cur.executemany(insert_sql, rows)
        self.conn.commit()
        
    def load_data_for_binary_classification(self, Loan_Status):
        sql = "select * from loan;"
        self.cur.execute(sql)
    
        data = self.cur.fetchall()
        
        self.X = [ (t['ApplicantIncome'], t['LoanAmount'] ) for t in data ]
        self.X = np.array(self.X)
        
        self.y = [ 1 if (t['Loan_Status'] == Loan_Status) else 0 for t in data]
        self.y = np.array(self.y)
    
    
    def data_split_train_test(self):
        self.X_train_input, self.X_test_input, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.3, random_state=42)
    
    def preprocess(self):
        ss=StandardScaler()
        ss.fit(self.X_train_input)
        ss.fit(self.X_test_input)
        self.X_train=ss.transform(self.X_train_input)
        self.X_test=ss.transform(self.X_test_input)
    
    def train_and_test_LogisticRegression(self):
        lr=LogisticRegression()
        lr_model=lr.fit(self.X_train, self.y_train)
        print(f"상위 5개의 predict={lr.predict(self.X_train[:5])}\n")
        print(f"상위 5개의 predict prob.={lr.predict_proba(self.X_train[:5])}\n")
        decisions_train=lr.decision_function(self.X_train[:5])
        print(f"상위 5개의 decision={decisions_train}\n")
        sigmoid_train=expit(decisions_train)
        print(f"상위 5개의 시그모이드 함수 통과={sigmoid_train}\n")

        lr_model.predict(self.X_test)
        decisions=lr.decision_function(self.X_test)
        self.y_predict=expit(decisions)
        pprint(f"test result={self.y_predict}\n")
        
    def classification_LR_performance_eval_binary(self, y_test, y_predict):
        tp, tn, fp, fn = 0,0,0,0
        
        for y, yp in zip(y_test, y_predict):
            if y == 1 and yp >= 0.5:
                tp += 1
            elif y == 1 and yp <= 0.5:
                fn += 1
            elif y == 0 and yp >= 0.5:
                fp += 1
            else:
                tn += 1
                
        accuracy = (tp+tn)/(tp+tn+fp+fn)
        precision = (tp)/(tp+fp)
        recall = (tp)/(tp+fn)
        f1_score = 2*precision*recall / (precision+recall)
        
        print("\n-----LR__eval_Iteration-----")
        print(f"accuracy={accuracy}")
        print(f"preceision={precision}")
        print(f"recall={recall}")
        print(f"f1_score={f1_score}\n")
        
        return accuracy, precision, recall, f1_score
    
    
    def train_and_test_RandomForest(self):
        rf = RandomForestClassifier(random_state= 42)
        rf_model = rf.fit(self.X_train, self.y_train)
        self.y_predict = rf_model.predict(self.X_test)
    
    def train_and_test_AdaBoost(self):
        ab = AdaBoostClassifier(random_state=42)
        ab_model = ab.fit(self.X_train, self.y_train)
        self.y_predict = ab_model.predict(self.X_test)
        
    def classification_performance_eval_binary(self, y_test, y_predict):
        tp, tn, fp, fn = 0,0,0,0
        
        for y, yp in zip(y_test, y_predict):
            if y == 1 and yp == 1:
                tp += 1
            elif y == 1 and yp == 0:
                fn += 1
            elif y == 0 and yp == 1:
                fp += 1
            else:
                tn += 1
                
        accuracy = (tp+tn)/(tp+tn+fp+fn)
        precision = (tp)/(tp+fp)
        recall = (tp)/(tp+fn)
        f1_score = 2*precision*recall / (precision+recall)
        
        '''
        print("\n-----train_test_split-----")
        print(f"accuracy={accuracy}")
        print(f"preceision={precision}")
        print(f"recall={recall}")
        print(f"f1_score={f1_score}\n")
        '''
        
        return accuracy, precision, recall, f1_score
    
    def binary_LogisticRegression_KFold_performance(self):
        
        kf = KFold(n_splits=5, random_state=42, shuffle=True)
        
        accuracy_scores = []
        precision_scores = []
        recall_scores = []
        f1_scores = []
        
        for train_index, test_index in kf.split(self.X):
            X_train, X_test = self.X[train_index], self.X[test_index]
            y_train, y_test = self.y[train_index], self.y[test_index]
            
            lr=LogisticRegression()
            lr_model=lr.fit(X_train, y_train)
            lr_model.predict(X_test)
            decisions=lr.decision_function(X_test)
            y_predict=expit(decisions)
            
            accuracy, precision, recall, f1_score=self.classification_LR_performance_eval_binary(y_test, y_predict)
            accuracy_scores.append(accuracy)
            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1_score)
        
        average_accuracy = np.mean(accuracy_scores)
        average_precision = np.mean(precision_scores)
        average_recall = np.mean(recall_scores)
        average_f1_score = np.mean(f1_scores)
        
        print("\nLR\nAverage Accuracy:", average_accuracy)
        print("Average Precision:", average_precision)
        print("Average Recall:", average_recall)
        print("Average F1 Score:", average_f1_score)
    
    def binary_RandomForest_KFold_performance(self):
        
        kf = KFold(n_splits=5, random_state=42, shuffle=True)
        
        accuracy_scores = []
        precision_scores = []
        recall_scores = []
        f1_scores = []
        
        for train_index, test_index in kf.split(self.X):
            X_train, X_test = self.X[train_index], self.X[test_index]
            y_train, y_test = self.y[train_index], self.y[test_index]
            
            rf = RandomForestClassifier(random_state= 42)
            rf_model = rf.fit(X_train, y_train)
            y_predict = rf_model.predict(X_test)
            
            accuracy, precision, recall, f1_score=self.classification_performance_eval_binary(y_test, y_predict)
            accuracy_scores.append(accuracy)
            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1_score)
        
        average_accuracy = np.mean(accuracy_scores)
        average_precision = np.mean(precision_scores)
        average_recall = np.mean(recall_scores)
        average_f1_score = np.mean(f1_scores)
        
        print("\nRF\nAverage Accuracy:", average_accuracy)
        print("Average Precision:", average_precision)
        print("Average Recall:", average_recall)
        print("Average F1 Score:", average_f1_score)
    
    def binary_AdaBoost_KFold_performance(self):
        
        kf = KFold(n_splits=5, random_state=42, shuffle=True)
        
        accuracy_scores = []
        precision_scores = []
        recall_scores = []
        f1_scores = []
        
        for train_index, test_index in kf.split(self.X):
            X_train, X_test = self.X[train_index], self.X[test_index]
            y_train, y_test = self.y[train_index], self.y[test_index]
            
            ab = AdaBoostClassifier(random_state=42)
            ab_model = ab.fit(X_train, y_train)
            y_predict = ab_model.predict(X_test)
            
            accuracy, precision, recall, f1_score=self.classification_performance_eval_binary(y_test, y_predict)
            accuracy_scores.append(accuracy)
            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1_score)
        
        average_accuracy = np.mean(accuracy_scores)
        average_precision = np.mean(precision_scores)
        average_recall = np.mean(recall_scores)
        average_f1_score = np.mean(f1_scores)
        
        print("\nAB\nAverage Accuracy:", average_accuracy)
        print("Average Precision:", average_precision)
        print("Average Recall:", average_recall)
        print("Average F1 Score:", average_f1_score)
        

def binary_LogisticRegression_train_test_performance():
    clf = class_loan_classification(import_data_flag=True)
    clf.load_data_for_binary_classification(Loan_Status='Y')
    clf.data_split_train_test()
    clf.preprocess()
    clf.train_and_test_LogisticRegression()
    clf.classification_LR_performance_eval_binary(clf.y_test, clf.y_predict)

def binary_RandomForest_train_test_performance():
    clf = class_loan_classification(import_data_flag=True)
    clf.load_data_for_binary_classification(Loan_Status='Y')
    clf.data_split_train_test()
    clf.preprocess()
    clf.train_and_test_RandomForest()
    clf.classification_performance_eval_binary(clf.y_test, clf.y_predict)
    
def binary_AdaBoost_train_test_performance():
    clf = class_loan_classification(import_data_flag=True)
    clf.load_data_for_binary_classification(Loan_Status='Y')
    clf.data_split_train_test()
    clf.preprocess()
    clf.train_and_test_AdaBoost()
    clf.classification_performance_eval_binary(clf.y_test, clf.y_predict)

def binary_LogisticRegression_KFold_performance():
    clf = class_loan_classification(import_data_flag=False)
    clf.load_data_for_binary_classification(Loan_Status='Y')
    print("\nKFold")
    clf.binary_LogisticRegression_KFold_performance()

def binary_RandomForest_KFold_performance():
    clf = class_loan_classification(import_data_flag=False)
    clf.load_data_for_binary_classification(Loan_Status='Y')
    clf.binary_RandomForest_KFold_performance()

def binary_AdaBoost_KFold_performance():
    clf = class_loan_classification(import_data_flag=False)
    clf.load_data_for_binary_classification(Loan_Status='Y')
    clf.binary_AdaBoost_KFold_performance()

if __name__ == "__main__":
    binary_LogisticRegression_train_test_performance()
    binary_RandomForest_train_test_performance()
    binary_AdaBoost_train_test_performance()
    
    binary_LogisticRegression_KFold_performance()
    binary_RandomForest_KFold_performance()
    binary_AdaBoost_KFold_performance()