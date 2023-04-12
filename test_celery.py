from temp import add
result = add.delay(4, 4)
print(result.wait())
