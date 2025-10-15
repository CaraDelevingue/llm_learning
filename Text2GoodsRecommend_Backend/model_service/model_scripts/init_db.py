import sys
import os

# 将根目录添加到 sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.crud import create_raw_table
from app.tests import test_connection

if __name__ == "__main__":
    #test_connection()
    create_raw_table()
    