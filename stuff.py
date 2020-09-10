def stuff(a, hash_table):
    if a[0] == 1:
        return
    a = [i - 1 for i in a]
    hash_table[a[0]] = a
    stuff(a, hash_table)
    return


a = [5, 3, 5]
print(stuff(a, {}))
