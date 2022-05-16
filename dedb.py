import sqlite3
conn=sqlite3.connect('test.db')
print("Opened database successfully")
conn.execute("DELETE from Company1 where Id=2")
conn.commit()
cursor=conn.execute("SELECT Id,name,age,address,salary from Company2")
for row in cursor:
    print "ID=",row[0]
    print "NAME=",row[1]
    print "ADDRESS=",row[3]
    print "Salary=",row[4],"\n"
print("Operation done successfully")
conn.close()
