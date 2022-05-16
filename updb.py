import sqlite3
conn=sqlite3.connect('test.db')
print("Opened database successfully")
conn.execute("UPDATE Company3 set salary=25000.00 where Id=1")
conn.commit()
print "Total no of rows updated:",conn.total_changes
cursor=conn.execute("SELECT Id,name,age,address,salary from Company2")
for row in cursor:
    print "ID=",row[0]
    print "NAME=",row[1]
    print "ADDRESS=",row[3]
    print "Salary=",row[4],"\n"
print("Operation done successfully")
conn.close()
