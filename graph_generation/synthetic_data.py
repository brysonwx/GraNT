import hashlib
import sqlite3
import pandas as pd
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
from functools import partial
from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError

"""
# DB operations
CREATE TABLE IF NOT EXISTS nodeidx2paperid (id INT AUTO_INCREMENT PRIMARY KEY, node_idx INT NOT NULL, paper_id VARCHAR(255) NOT NULL);
CREATE TABLE IF NOT EXISTS titleabs (id INT AUTO_INCREMENT PRIMARY KEY, paper_id VARCHAR(255) NOT NULL, title_id CHAR(32) NOT NULL);
CREATE TABLE IF NOT EXISTS oag (id INT AUTO_INCREMENT PRIMARY KEY, oag_paper_id VARCHAR(255) NOT NULL, n_citation INT NOT NULL, title_id CHAR(32) NOT NULL);

ALTER TABLE nodeidx2paperid ADD INDEX suoyin_paper_id (paper_id);
ALTER TABLE titleabs ADD INDEX suoyin_paper_id (paper_id);
ALTER TABLE nodeidx2paperid ADD title_id_join CHAR(32) NOT NULL;
UPDATE nodeidx2paperid LEFT JOIN titleabs ON nodeidx2paperid.paper_id = titleabs.paper_id SET nodeidx2paperid.title_id_join = titleabs.title_id;

ALTER TABLE nodeidx2paperid ADD COLUMN n_citation INT DEFAULT NULL;
ALTER TABLE nodeidx2paperid ADD COLUMN oag_paper_id VARCHAR(255) DEFAULT NULL;

ALTER TABLE nodeidx2paperid ADD INDEX suoyin_title_id_join (title_id_join);
ALTER TABLE oag ADD INDEX suoyin_title_id (title_id);
UPDATE nodeidx2paperid
LEFT JOIN oag ON nodeidx2paperid.title_id_join = oag.title_id
SET
    nodeidx2paperid.oag_paper_id = oag.oag_paper_id,
    nodeidx2paperid.n_citation = oag.n_citation;
"""


def get_mysql_engine(MYSQL_USER, MYSQL_PASSWORD, MYSQL_HOST, MYSQL_DATABASE):
    engine = create_engine(f'mysql+mysqlconnector://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}/{MYSQL_DATABASE}')
    return engine


def generate_title_id(partition_df):
    partition_df = partition_df.copy()
    partition_df['title_id'] = partition_df['title'].apply(
        lambda x: hashlib.md5(x.lower().strip().encode('utf-8')).hexdigest() if isinstance(x, str) else None
    )
    partition_df = partition_df.drop(columns=['title'])
    return partition_df


def batch_write_to_mysql(partition_df, db_engine, db_name):
    if isinstance(partition_df, dd.DataFrame):
        partition_df = partition_df.compute()
    # partition_df.to_sql('oag', con=engine, if_exists='append', index=False)
    try:
        partition_df.to_sql(db_name, con=db_engine, if_exists='append', index=False, method='multi')
    except IntegrityError as e:
        print(f"Error inserting data: {e}")
    return partition_df


def handle_oag_data(oag_file_path, db_engine, db_name):
    fields_to_extract = ['id', 'n_citation', 'title']
    df = dd.read_json(oag_file_path, lines=True, blocksize="128MB")
    df_selected = df[fields_to_extract].rename(columns={'id': 'oag_paper_id'})
    df_selected = df_selected[df_selected['n_citation'] > 0]
    df_selected = df_selected.map_partitions(
        generate_title_id,
        meta={'oag_paper_id': 'object', 'n_citation': 'int64', 'title_id': 'object'}
    )

    # df_selected = df_selected.drop_duplicates(subset='title_id')

    write_func = partial(batch_write_to_mysql, db_engine=db_engine, db_name=db_name)
    with ProgressBar():
        df_selected.map_partitions(write_func, meta=df_selected._meta).compute()


def handle_nodeidx2paperid_data(file_path, db_engine, db_name):
    fields_to_extract = ['node idx', 'paper id']
    renamed_fields = {'node idx': 'node_idx', 'paper id': 'paper_id'}

    df = dd.read_csv(file_path, blocksize="128MB", usecols=fields_to_extract)
    df_selected = df.rename(columns=renamed_fields)
    write_func = partial(batch_write_to_mysql, db_engine=db_engine, db_name=db_name)
    with ProgressBar():
        df_selected.map_partitions(write_func, meta=df_selected._meta).compute()


def handle_titleabs_data(file_path, db_engine, db_name):
    column_names = ['paper_id', 'title', 'abstract']
    fields_to_extract = ['paper_id', 'title']
    df = dd.read_csv(file_path, sep='\t', names=column_names, blocksize="128MB")
    df_selected = df[fields_to_extract]
    df_selected = df_selected.map_partitions(
        generate_title_id,
        meta={'paper_id': 'object', 'title_id': 'object'}
    )

    write_func = partial(batch_write_to_mysql, db_engine=db_engine, db_name=db_name)
    with ProgressBar():
        df_selected.map_partitions(write_func, meta=df_selected._meta).compute()


"""
import the nodeidx2paperid.sql into sqlite3 memory database
1) export the data table: [nodeidx2paperid] from mysql database: [grant_data]
mysqldump -u root -p -h localhost grant_data nodeidx2paperid > nodeidx2paperid.sql
2) modify the content of nodeidx2paperid.sql to meet the requirements of sqlite3
open the nodeidx2paperid.sql with vscode and modify the content.
the original content: 
CREATE TABLE `nodeidx2paperid` (
  `id` int NOT NULL AUTO_INCREMENT,
  `node_idx` int NOT NULL,
  `paper_id` varchar(255) NOT NULL,
  `title_id_join` char(32) NOT NULL,
  `n_citation` int DEFAULT NULL,
  `oag_paper_id` varchar(255) DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `suoyin_paper_id` (`paper_id`),
  KEY `suoyin_title_id_join` (`title_id_join`)
) ENGINE=InnoDB AUTO_INCREMENT=169344 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

the modified content: 
CREATE TABLE `nodeidx2paperid` (
  `id` INTEGER PRIMARY KEY,
  `node_idx` INTEGER NOT NULL,
  `paper_id` TEXT NOT NULL,
  `title_id_join` TEXT NOT NULL,
  `n_citation` INTEGER DEFAULT NULL,
  `oag_paper_id` TEXT DEFAULT NULL
);
CREATE INDEX suoyin_paper_id ON nodeidx2paperid (paper_id);
CREATE INDEX suoyin_title_id_join ON nodeidx2paperid (title_id_join);

the original content:
LOCK TABLES `nodeidx2paperid` WRITE;
balabala...
UNLOCK TABLES;

the modified content:
BEGIN TRANSACTION;
balabala...(no modification)
COMMIT;
"""


def get_nodeidx_citation(sql_file: str = None):
    if not sql_file:
        sql_file = 'nodeidx2paperid.sql'
    with open(sql_file, 'r') as f:
        sql_script = f.read()
    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()
    cursor.executescript(sql_script)
    query = "SELECT DISTINCT node_idx, n_citation FROM nodeidx2paperid"
    df = pd.read_sql_query(query, conn)
    node_to_citation = dict(zip(df['node_idx'], df['n_citation']))
    conn.close()
    return node_to_citation


if __name__ == '__main__':
    MYSQL_USER = 'root'
    MYSQL_PASSWORD = 'grant_data'
    MYSQL_HOST = 'localhost'
    MYSQL_DATABASE = 'grant_data'
    engine = get_mysql_engine(MYSQL_USER, MYSQL_PASSWORD, MYSQL_HOST, MYSQL_DATABASE)

    # 1. see: https://www.aminer.cn/open/article?id=5965cf249ed5db41ed4f52bf,
    # download the oag files
    oag_json_file_paths = ['~/oag_files/v3.1_oag_publication_{}.json'.format(i) for i in range(1, 15)]
    for file_path in oag_json_file_paths:
        handle_oag_data(file_path, engine, 'oag')

    # 2. [Pytorch Goemetric download]
    # see your dataset directory such as OGB/ogbn_arxiv/mapping, find the nodeidx2paperid.csv.gz
    nodeidx2paperid_file_path = '~/nodeidx2paperid.csv'
    handle_nodeidx2paperid_data(nodeidx2paperid_file_path, engine, 'nodeidx2paperid')

    # 3. see: https://ogb.stanford.edu/docs/nodeprop/#ogbn-arxiv,
    # download the titleabs.tsv
    titleabs_file_path = '~/titleabs.tsv'
    handle_titleabs_data(titleabs_file_path, engine, 'titleabs')

    # test the logic
    # get_nodeidx_citation('nodeidx2paperid.sql')
