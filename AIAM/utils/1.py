import pandas as pd
import psycopg2
from psycopg2 import OperationalError

# 连接数据库
def connect_to_db(host, dbname, user, password, port):
    try:
        conn = psycopg2.connect(
            host=host,
            dbname=dbname,
            user=user,
            password=password,
            port=port
        )
        return conn
    except OperationalError as e:
        print(f"Error: {e}")
        return None

# 导入CSV文件到数据库表
def import_csv_to_db(conn, csv_file, table_name):
    try:
        # 读取CSV文件
        df = pd.read_csv(csv_file)

        # 创建一个游标对象
        cursor = conn.cursor()

        # 构建INSERT INTO语句
        insert_query = f"INSERT INTO {table_name} ({', '.join(df.columns)}) VALUES %s"

        # 将DataFrame转换为元组列表
        values = [tuple(row) for row in df.values]

        # 执行插入操作
        cursor.executemany(insert_query, values)
        conn.commit()
        cursor.close()

        print(f"Successfully imported {csv_file} into {table_name}")
    except Exception as e:
        print(f"Error: {e}")

# 主函数
def main():
    # 数据库连接信息
    host = '192.168.1.19'
    dbname = '123'
    user = 'demo'
    password = '0147258'
    port = '5432'  # 默认是5432

    # CSV 文件路径
    csv_file = 'F:\\testdata'

    # 目标表名
    table_name = '1'

    # 连接到数据库
    conn = connect_to_db(host, dbname, user, password, port)

    if conn:
        # 导入CSV文件到数据库
        import_csv_to_db(conn, csv_file, table_name)
        conn.close()
    else:
        print("Failed to connect to the database.")

if __name__ == "__main__":
    main()
