def binary_search(arr,low,high,x):
    if high>=low:
        mid=(high+low)//2
        if arr[mid]==x:
            return mid
        elif arr[mid]>x:
            return binary_search(arr,low,mid-1,x)
        else:
            return binary_search(arr,mid+1,high,x)
    else:
        return -1
arr=[2,31,42,11,48]
x=4
result=binary_search(arr,0,len(arr)-1,x)
if result!=-1:
 print('Element is present at index',str(result))
else:
 print('Element is not present in array')
       
def binary_search(arr,x):
       low=0
       high=len(arr)-1
       mid=0
       while low<=high:
         mid=(high+low)//2
         if arr[mid]<x:
           low=mid+1
         elif arr[mid]>x:
           high=mid-1
         else:
             return mid
       return -1
arr=[9,4,7,15,42,68,33]
x=10
result=binary_search(arr,x)
if result!=-1:
       print('Element is present at index',str(result))
else:
       print('Element is not present in array')
