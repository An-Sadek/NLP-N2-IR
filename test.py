import re

pattern = r"\S*\d+\S*"
test = "1234, 54321, A1234, 1234B. Thanh cong"

result = re.sub(pattern, "", test)
print(result)