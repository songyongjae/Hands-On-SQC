import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from db_conn import *
from pprint import pprint
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, confusion_matrix

class class_wine_classification():
    def __init__(self, import_data_flag=True):
        self.conn, self.cur = open_db()
        if import_data_flag:
            self.import_wine_data()
    
    def import_wine_data(self):
        drop_sql =""" drop table if exists wine;"""
        self.cur.execute(drop_sql)
        self.conn.commit()
    
        create_sql = """
            create table wine (
                id int auto_increment primary key,
                Class int,
                Alcohol float,
                Malic_acid float,
                Ash float,
                Alcalinity_of_ash float,
                Magnesium float,
                Total_phenols float,
                Flavanoids float,
                Nonflavanoid_phenols float,
                Proanthocyanins float,
                Color_intensity float,
                Hue float,
                OD280_OD315_of_diluted_wines float,
                Proline float
                ); 
        """
    
        self.cur.execute(create_sql)
        self.conn.commit()
    
        file_name = './data/wine.data.csv'
        wine_data = pd.read_csv(file_name)

        rows = []
    
        insert_sql = """insert into wine(Class,Alcohol,Malic_acid,Ash,Alcalinity_of_ash,Magnesium,Total_phenols,Flavanoids,Nonflavanoid_phenols,Proanthocyanins,Color_intensity,Hue,OD280_OD315_of_diluted_wines,Proline)
                        values(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s);"""
    
        for t in wine_data.values:
            rows.append(tuple(t))
    
        self.cur.executemany(insert_sql, rows)
        self.conn.commit()
    
    def load_data_for_multiclass_classification(self):
        sql = "select * from wine;"
        self.cur.execute(sql)
    
        data = self.cur.fetchall()
        
        self.X = [ (t['Alcohol'], t['Malic_acid'],t['Ash'],t['Alcalinity_of_ash'],
                    t['Magnesium'],t['Total_phenols'],t['Flavanoids'],t['Nonflavanoid_phenols'],
                    t['Proanthocyanins'],t['Color_intensity'],t['Hue'],
                    t['OD280_OD315_of_diluted_wines'],t['Proline']) for t in data ]
        self.X = np.array(self.X)
        
        self.y =  [t['Class'] for t in data]
        self.y = np.array(self.y)
        
    def data_split_train_test(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.3, random_state=42)
        
    def classification_performance_eval_multiclass(self, y_test, y_predict, output_dict=False):
        target_names=['C1', 'C2', 'C3']
        labels = [1,2,3]
        self.classification_report = classification_report(y_test, y_predict, target_names=target_names, labels=labels, output_dict=output_dict)
        self.confusion_matrix = confusion_matrix(y_test, y_predict, labels=labels)
        #pprint(f"[classification_report]\n{self.classification_report}")
        pprint(f"[confusion_matrix]\n{self.confusion_matrix}")


    def train_and_test_GB(self):
        gb = GradientBoostingClassifier(random_state= 42)
        gb_model = gb.fit(self.X_train, self.y_train)
        self.y_predict = gb_model.predict(self.X_test)
        print(f"GB : self.y_predict[:10]={self.y_predict[:10]}")
        print(f"GB : self.y_test[:10]={self.y_test[:10]}")
    
    def train_and_test_RandomForest(self):
        rf=RandomForestClassifier(random_state= 42)
        rf_model=rf.fit(self.X_train, self.y_train)
        self.y_predict = rf_model.predict(self.X_test)
        print(f"\nRF : self.y_predict[:10]={self.y_predict[:10]}")
        print(f"RF : self.y_test[:10]={self.y_test[:10]}")
        
    def train_and_test_AdaBoost(self):
        ab = AdaBoostClassifier(random_state=42)
        ab_model = ab.fit(self.X_train, self.y_train)
        self.y_predict = ab_model.predict(self.X_test)
        print(f"\nAB : self.y_predict[:10]={self.y_predict[:10]}")
        print(f"AB : self.y_test[:10]={self.y_test[:10]}")
    
    
    
    def multiclass_GB_KFold_performance(self):
        print("\n\n\nGB_KFold\n")
        kfold_reports = []
        kf = KFold(n_splits=5, random_state=42, shuffle=True)
        for train_index, test_index in kf.split(self.X):
            X_train, X_test = self.X[train_index], self.X[test_index]
            y_train, y_test = self.y[train_index], self.y[test_index]
            
            gb = GradientBoostingClassifier(random_state= 42)
            gb_model = gb.fit(X_train, y_train)
            y_predict = gb_model.predict(X_test)
            
            self.classification_performance_eval_multiclass(y_test, y_predict, output_dict=True)
            kfold_reports.append(pd.DataFrame(self.classification_report).transpose())
            
        for s in kfold_reports:
            print('\n_', s)
            
        mean_report = pd.concat(kfold_reports).groupby(level=0).mean()
        print('\n\nGB result:\nmean\n', mean_report)
        
    def multiclass_RF_KFold_performance(self):
        print("\nRF_KFold\n")
        kfold_reports = []
        kf = KFold(n_splits=5, random_state=42, shuffle=True)
        for train_index, test_index in kf.split(self.X):
            X_train, X_test = self.X[train_index], self.X[test_index]
            y_train, y_test = self.y[train_index], self.y[test_index]
            
            rf = RandomForestClassifier(random_state= 42)
            rf_model = rf.fit(X_train, y_train)
            y_predict = rf_model.predict(X_test)
            self.classification_performance_eval_multiclass(y_test, y_predict, output_dict=True)
            kfold_reports.append(pd.DataFrame(self.classification_report).transpose())
            
        for s in kfold_reports:
            print('\n_', s)
            
        mean_report = pd.concat(kfold_reports).groupby(level=0).mean()
        print('\nRF result:\nmean\n', mean_report)
    
    def multiclass_AB_KFold_performance(self):
        print("\nAB_KFold\n")
        kfold_reports = []
        kf = KFold(n_splits=5, random_state=42, shuffle=True)
        for train_index, test_index in kf.split(self.X):
            X_train, X_test = self.X[train_index], self.X[test_index]
            y_train, y_test = self.y[train_index], self.y[test_index]
            
            ab = AdaBoostClassifier(random_state=42)
            ab_model = ab.fit(X_train, y_train)
            y_predict = ab_model.predict(X_test)
            self.classification_performance_eval_multiclass(y_test, y_predict, output_dict=True)
            kfold_reports.append(pd.DataFrame(self.classification_report).transpose())
            
        for s in kfold_reports:
            print('\n_', s)
            
        mean_report = pd.concat(kfold_reports).groupby(level=0).mean()
        print('\nAB result:\nmean\n', mean_report)


def multiclass_GB_train_test_performance():
    clf = class_wine_classification(import_data_flag=True)
    clf.load_data_for_multiclass_classification()
    clf.data_split_train_test()
    clf.train_and_test_GB()
    clf.classification_performance_eval_multiclass(clf.y_test, clf.y_predict)
    
def multiclass_RandomForest_train_test_performance():
    clf = class_wine_classification(import_data_flag=True)
    clf.load_data_for_multiclass_classification()
    clf.data_split_train_test()
    clf.train_and_test_RandomForest()
    clf.classification_performance_eval_multiclass(clf.y_test, clf.y_predict)

def multiclass_AdaBoost_train_test_performance():
    clf = class_wine_classification(import_data_flag=True)
    clf.load_data_for_multiclass_classification()
    clf.data_split_train_test()
    clf.train_and_test_AdaBoost()
    clf.classification_performance_eval_multiclass(clf.y_test, clf.y_predict)
    
def multiclass_GB_KFold_performance_class():
    clf = class_wine_classification(import_data_flag=True)
    clf.load_data_for_multiclass_classification()
    clf.multiclass_GB_KFold_performance()

def multiclass_RF_KFold_performance_class():
    clf = class_wine_classification(import_data_flag=True)
    clf.load_data_for_multiclass_classification()
    clf.multiclass_RF_KFold_performance()

def multiclass_AB_KFold_performance_class():
    clf = class_wine_classification(import_data_flag=True)
    clf.load_data_for_multiclass_classification()
    clf.multiclass_AB_KFold_performance()

if __name__ == "__main__":
    multiclass_GB_train_test_performance()
    multiclass_RandomForest_train_test_performance()
    multiclass_AdaBoost_train_test_performance()
    
    multiclass_GB_KFold_performance_class()
    multiclass_RF_KFold_performance_class()
    multiclass_AB_KFold_performance_class()