from langchain_community.utilities import SQLDatabase

db = SQLDatabase.from_uri("sqlite:///Chinook.db")

print("Todas as tabelas desse banco de teste:\n" + str(db.get_usable_table_names()))
