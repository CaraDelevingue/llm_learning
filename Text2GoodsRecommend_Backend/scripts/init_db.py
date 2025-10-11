import sys
import os

# 将 backend 添加到 sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database.setup import create_raw_table,test

if __name__ == "__main__":
    create_raw_table()
    #test()