from sqlalchemy import create_engine

SERVER = "localhost\\SQLEXPRESS"
DATABASE = "bank_reviews"

engine = create_engine(
    "mssql+pyodbc://@localhost\\SQLEXPRESS/"
    + DATABASE +
    "?driver=ODBC+Driver+17+for+SQL+Server&trusted_connection=yes"
)

print("Database connection successful")