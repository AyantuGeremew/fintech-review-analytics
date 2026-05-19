from sqlalchemy import create_engine

engine = create_engine(
    "mssql+pyodbc://@localhost\\SQLEXPRESS/bank_reviews"
    "?driver=ODBC+Driver+17+for+SQL+Server&trusted_connection=yes"
)

print(engine.execute("SELECT COUNT(*) FROM Reviews").fetchall())