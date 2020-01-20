from sys import argv

script, file_name = argv

a = f"{str(file_name} delete the file"

print(a+'delete the file')
print("취소하려면 CTRL-C (^C) 를 누르세요.")
print("진행하려면 리턴 키를 누르세요.")

input("?")

print("파일 여는 중...")
dest = open(file_name, 'w', encoding='utf-8')

print("Delete the file contents. Bye!")
dest.truncate()

print("이제 세 줄에 들어갈 내용을 부탁드릴게요.")

line1 = input("1 Line: ")
line2 = input("2 Line: ")
line3 = input("3 Line: ")

print("이 내용을 파일에 씁니다.")

dest.write(line1)
dest.write("\n")
dest.write(line2)
dest.write("\n")
dest.write(line3)
dest.write("\n")

print("Close the file, finally")
dest.close()