import sqlite3
conn=sqlite3.connect('test.db')
print("Opened database successfully")
cursor=conn.execute("SELECT Id,name,age,address,salary from Company3")
for row in cursor:
    print "ID=",row[0]
    print "NAME=",row[1]
    print "ADDRESS=",row[3]
    print "Salary=",row[4],"\n"
print("Operation done successfully")
conn.close()
