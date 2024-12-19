import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from db_conn import *
from konlpy.tag import * 
import matplotlib.pyplot as plt
import os

class class_document_tfidf():
    def __init__(self):
        self.conn, self.cur = open_db()
        self.news_article_excel_file = './data/combined_article1.xlsx'
        self.pos_tagger = Kkma()
    
    def combine_excel_file(self):
        directory_path = './'
        
        excel_files = [file for file in os.listdir(directory_path) if file.endswith('.xlsx')]
        
        combined_df = pd.DataFrame()
        for file in excel_files:
            try:
                file_path = os.path.join(directory_path, file)
                df = pd.read_excel(file_path)
                df = df[['url', 'title', 'content']]
                combined_df = pd.concat([combined_df, df], ignore_index=True)
            except Exception as e:
                print(file_path)
                print(e)
                continue
    
        combined_df.to_excel(self.news_article_excel_file, index=False)   
    
    
    def import_news_article(self):
        drop_sql =""" drop table if exists news_article1;"""
        self.cur.execute(drop_sql)
        self.conn.commit()
    
        create_sql = """            
            CREATE TABLE news_article1 (
              id int auto_increment primary key,
              url varchar(500),
              title varchar(500),
              content TEXT,
              enter_date datetime default now()
            ) ;
        """

        self.cur.execute(create_sql)
        self.conn.commit()
    
        file_name = self.news_article_excel_file
        news_article_data = pd.read_excel(file_name)
    
        rows = []
    
        insert_sql = """insert into news_article1(url, title, content)
                        values(%s,%s,%s);"""
    
        for _, t in news_article_data.iterrows():
            t = tuple(t)
            try:
                self.cur.execute(insert_sql, t)
            except:
                continue
            rows.append(tuple(t))
    
        self.cur.executemany(insert_sql, rows)
        self.conn.commit()
        print("table created and data loaded\n")
        
    def extract_nouns(self):
        
        create_sql = """
            drop table if exists extracted_terms1;
            drop table if exists term_dict1;
            
            create table extracted_terms1 (
                id int auto_increment primary key,
                doc_id int,
                term varchar(30),
                term_region varchar(10),
                seq_no int,
                enter_date datetime default now()    ,
                index(term)
                );
            
            create table term_dict1 (
                id int auto_increment primary key,
                term varchar(30),
                enter_date datetime default now(),
                index(term)
                );
        """
        
        self.cur.execute(create_sql)
        self.conn.commit()
        
        sql = """select * from news_article1;"""
        self.cur.execute(sql)
        
        r = self.cur.fetchone()
        
        noun_terms = set()
        
        rows = []
        
        while r:
            print(f"doc_id={r['id']}")
            i = 0
            title_res = self.pos_tagger.nouns(r['title'])
            content_res = self.pos_tagger.nouns(r['content'])
            
            title_rows = [ (r['id'], t, 'title', i+1) for i, t in enumerate(title_res) ]
            content_rows = [ (r['id'], c, 'content', i+1) for i, c in enumerate(content_res) ]
            
            rows += title_rows
            rows += content_rows

            print(f"title_res={title_res}")
            print(f"content_res={content_res}")
            
            noun_terms.update(title_res)
            noun_terms.update(content_res)
            
            r = self.cur.fetchone()
            
        if rows:
            insert_sql = """insert into extracted_terms1(doc_id, term, term_region, seq_no)
                            values(%s,%s,%s,%s);"""
            self.cur.executemany(insert_sql, rows)
            self.conn.commit()

        print(noun_terms)
        print(f"\nnumber of terms = {len(noun_terms)}")
        
        insert_sql = """insert into term_dict1(term) values (%s);"""
        self.cur.executemany(insert_sql, list(noun_terms))
        self.conn.commit()
    
    def gen_idf(self):

        create_sql = """
            drop table if exists idf1;
            
            create table idf1 (
                term_id int primary key,
                df int,
                idf float,
                enter_date datetime default now()                
                );
        """
        
        self.cur.execute(create_sql)
        self.conn.commit()
        
        sql = "select count(*) as doc_count from news_article1;"
        self.cur.execute(sql)
        self.num_of_docs = self.cur.fetchone()['doc_count']
        
        idf_sql = f""" insert into idf1(term_id, df, idf)
                select ntd.id, count(distinct doc_id) as df, log({self.num_of_docs}/count(distinct doc_id)) as idf
                from extracted_terms1 ent, term_dict1 ntd
                where ent.term = ntd.term
                group by ntd.id;
            """
        self.cur.execute(idf_sql)        
        self.conn.commit()


    def show_top_df_terms(self):
        sql = """select * from idf1 idf, term_dict1 td
                where idf.term_id = td.id
                order by df desc
                ;"""
        self.cur.execute(sql)
        
        res = [ (r['term'], r['df'], r['idf']) for r in self.cur.fetchall() ]

        print("\nTop 10 DF terms:\n")
        
        for r in res[:10]:
            print(f"{r[0]}: df={r[1]}, idf={r[2]}")
        
        df_list = [ r[1] for r in res]
        plt.figure(figsize=(10, 5))
        plt.hist(df_list, bins=100, alpha=0.7, color='blue') 
        plt.title('Histogram of DF')
        plt.xlabel('Document Frequency')
        plt.ylabel('Number of Terms')
        plt.grid(axis='y', alpha=0.75)
        plt.show()        


    def show_top_idf_terms(self):
        sql = """select * from idf1 idf, term_dict1 td
                where idf.term_id = td.id
                order by idf desc
                ;"""
        self.cur.execute(sql)
        
        res = [ (r['term'], r['df'], r['idf']) for r in self.cur.fetchall() ]

        print("\nTop 10 IDF terms:\n")
        
        for r in res[:10]:
            print(f"{r[0]}: df={r[1]}, idf={r[2]}")

        idf_list = [ r[2] for r in res]
        plt.figure(figsize=(10, 5))
        plt.hist(idf_list, bins=100, alpha=0.7, color='red') 
        plt.title('Histogram of IDF')
        plt.xlabel('IDF')
        plt.ylabel('Number of Terms')
        plt.grid(axis='y', alpha=0.75)
        plt.show()  


    def gen_tfidf(self):
        
        create_sql = """
            drop table if exists tfidf1;
            
            create table tfidf1 (
                id int auto_increment primary key,
                doc_id int,
                term_id int,
                tf float,
                tfidf float,
                enter_date datetime default now()
                );
        """
        
        self.cur.execute(create_sql)
        self.conn.commit()
        

        tfidf_sql = """ insert into tfidf1(doc_id, term_id, tf, tfidf )  
                        select ent.doc_id, ntd.id, count(*) as tf, count(*) * idf.idf as tfidf
                        from extracted_terms1 ent, term_dict1 ntd, idf1 idf
                        where ent.term = ntd.term and ntd.id = idf.term_id
                        group by ent.doc_id, ntd.id;
                    """

        self.cur.execute(tfidf_sql)        
        self.conn.commit()

    def get_keywords_of_document(self, doc):
        sql = f""" 
            select *
            from tfidf1 tfidf, term_dict1 td
            where tfidf.doc_id = {doc}
            and tfidf.term_id = td.id
            order by tfidf.tfidf desc
            limit 5;
        """

        self.cur.execute(sql)
        
        result = self.cur.fetchall()
        
        print(f"\nTop 5 키워드 for 문서 {doc}:")
        for row in result:
            print(f"{row['term']}: {row['tfidf']}")
    
    def sort_similar_docs(self, doc):
        sim_vector = []

        for i in range(1,1451):
            if i == doc:
                continue
            sim = self.doc_similarity(doc, i)
            sim_vector.append((i, sim))
        sorted_sim_vector = sorted(sim_vector, key=lambda x: x[1], reverse=True)
        
        print(f"\n문서 {doc}와 가장 가까운 문서 Top 3:")
        for idx, similarity in sorted_sim_vector[:3]:
            print(f"문서 {idx}: 유사도 {similarity}")
    
    
    def doc_similarity(self, doc1, doc2):
        def cosine_similarity(vec1, vec2):

            dict1 = dict(vec1)
            dict2 = dict(vec2)
            
            common_terms = set(dict1.keys()) & set(dict2.keys())
            dot_product = sum([dict1[term] * dict2[term] for term in common_terms])
            
            vec1_magnitude = sum([val**2 for val in dict1.values()])**0.5
            vec2_magnitude = sum([val**2 for val in dict2.values()])**0.5
            
            if vec1_magnitude == 0 or vec2_magnitude == 0:
                return 0
            else:
                return dot_product / (vec1_magnitude * vec2_magnitude)        
        
        sql1 = f"""select term_id, tfidf from tfidf1 where doc_id = {doc1};"""
        self.cur.execute(sql1)
        doc1_vector = [(t['term_id'], t['tfidf']) for t in self.cur.fetchall()]
        
        sql2 = f"""select term_id, tfidf from tfidf1 where doc_id = {doc2};"""
        self.cur.execute(sql2)
        doc2_vector = [(t['term_id'], t['tfidf']) for t in self.cur.fetchall()]
    
        return cosine_similarity(doc1_vector, doc2_vector)
    
    
    #Q. 사용자의 쿼리를 입력으로 받아,이 쿼리에 적합한 상위문서 top3를 보여줄 것
    ############################################################
    def process_user_query(self, user_query):
        def calculate_tf(document, words):
            tf = {}
            for w in words:
                tf[w] = document.count(w)
            return list(tf.values())
        
        drop_sql1 =""" drop table if exists query1;"""
        self.cur.execute(drop_sql1)
        self.conn.commit()
    
        create_sql1 = """            
            CREATE TABLE query1 (
              id int auto_increment primary key,
              term varchar(30),
              tf float
            ) ;
        """
        self.cur.execute(create_sql1)
        self.conn.commit()
        
        drop_sql2 =""" drop table if exists query_tfidf1;"""
        self.cur.execute(drop_sql2)
        self.conn.commit()
    
        create_sql2 = """
            CREATE TABLE query_tfidf1 (
              id int auto_increment primary key,
              term_id int,
              tfidf float
            ) ;
        """
        self.cur.execute(create_sql2)
        self.conn.commit()
        
        rows = []
        query_terms = self.pos_tagger.nouns(user_query)
        tf_values = calculate_tf(user_query, query_terms)
        query_rows = list(zip(query_terms, tf_values))
        rows += query_rows
        
        if rows:
            insert_sql = """insert into query1(term, tf)
                            values(%s, %s);"""
            self.cur.executemany(insert_sql, rows)
            self.conn.commit()
        
        tfidf_sql = """ insert into query_tfidf1(term_id, tfidf )
                        select ntd.id, AVG(que.tf * idf.idf) as tfidf
                        from extracted_terms ent, query1 que, idf1 idf, term_dict1 ntd
                        where ent.term = que.term and que.term = ntd.term and ntd.id = idf.term_id
                        group by ntd.id
                    """

        self.cur.execute(tfidf_sql)        
        self.conn.commit()
        
        sql2 = f"""select term_id, tfidf from query_tfidf1;"""
        self.cur.execute(sql2)
        query_vector = [(t['term_id'], t['tfidf']) for t in self.cur.fetchall()]
        
        print(f"\n쿼리에 대한 상위 문서 Top 3:")
        self.sort_similar_docs2(query_vector)
    
    def sort_similar_docs2(self, query_vector):
        sim_vector = []

        for i in range(1,1451):
            sim = self.doc_similarity2(query_vector, i)
            sim_vector.append((i, sim))
        sorted_sim_vector = sorted(sim_vector, key=lambda x: x[1], reverse=True)
        
        print(f"\nQuery와 가장 가까운 문서 Top 3:")
        for idx, similarity in sorted_sim_vector[:3]:
            print(f"문서 {idx}: 유사도 {similarity}")
    
    def doc_similarity2(self, query_vector, doc):
        def cosine_similarity(vec1, vec2):

            dict1 = dict(vec1)
            dict2 = dict(vec2)
            
            common_terms = set(dict1.keys()) & set(dict2.keys())
            dot_product = sum([dict1[term] * dict2[term] for term in common_terms])
            
            vec1_magnitude = sum([val**2 for val in dict1.values()])**0.5
            vec2_magnitude = sum([val**2 for val in dict2.values()])**0.5
            
            if vec1_magnitude == 0 or vec2_magnitude == 0:
                return 0
            else:
                return dot_product / (vec1_magnitude * vec2_magnitude)        
        
        doc1_vector = query_vector

        sql2 = f"""select term_id, tfidf from tfidf1 where doc_id = {doc};"""
        self.cur.execute(sql2)
        doc2_vector = [(t['term_id'], t['tfidf']) for t in self.cur.fetchall()]
        
        return cosine_similarity(doc1_vector, doc2_vector)

if __name__ == '__main__':
    cdb = class_document_tfidf()
    cdb.combine_excel_file()
    cdb.import_news_article()
    cdb.extract_nouns()
    
    cdb.gen_idf()
    cdb.show_top_df_terms()
    cdb.show_top_idf_terms()
    cdb.gen_tfidf()
    
    cdb.get_keywords_of_document(300)
    cdb.sort_similar_docs(80)
    cdb.process_user_query('공매도 금지 이후 에코프로 등 증시 상승을 주도한 이차전지 관련주도 공매도 금지 첫날만 상한가로 치솟았다가 도로 하락했다')