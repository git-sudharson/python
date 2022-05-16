import sqlite3
conn=sqlite3.connect('test.db')
print("Opened database successfully")
conn.execute("Insert into Company3(Id,name,age,address,salary) values(1,'Paul',32,'California',2000.00)")
conn.execute("Insert into Company3(Id,name,age,address,salary) values(2,'henry',33,'berlin',3000.00)")
conn.execute("Insert into Company3(Id,name,age,address,salary) values(3,'siraj',26,'tokyo',4000.00)")
conn.commit()
print("Records created successfully")
conn.close()
