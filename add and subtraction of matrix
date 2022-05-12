# Python program to add two matrices user input

r = int(input("Enter the rows: "))
c = int(input("Enter the columns: "))

print("Enter Matrix 1:")
m1 = [[int(input()) for i in range(c)] for i in range(r)]
print("Matrix 1 is: ")
for n in m1:
   print(n)

print("Enter Matrix 2:")
m2 = [[int(input()) for i in range(c)] for i in range(r)]
for n in m2:
   print(n)

r = [[0 for i in range(c)] for i in range(r)]
print("1)addition ")
print("2)subtraction ")
choice=int(input("Enter your choice : "))
if choice==1:     
    for i in range(len(r)):
        for j in range(c):
             r[i][j] = [m1[i][j] + m2[i][j]]
elif choice==2:
        
    for i in range(len(r)):
        for j in range(c):
             r[i][j] = [m1[i][j] - m2[i][j]]
else:
   print("Please enter the valid choice ")
if choice==1 or choice==2:
   print("Resultant Matrix:")
   for i in r:
      print(i)
