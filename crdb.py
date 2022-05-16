import sqlite3
conn=sqlite3.connect('test.db')
print("Opened databased successfully")
conn.execute('''create table Company3
             (Id int primary key not null,
             name Text not null,
             age int not null,
             address  char(50),
             salary Real)''')
print("Table created successfully")
conn.close()
