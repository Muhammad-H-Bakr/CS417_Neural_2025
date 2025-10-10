def list_sum(ls, number):
    a = []
    for x in range(0, len(ls)):
        for y in range(x + 1, len(ls)):
            if ls[x] + ls[y] == number:
                a.append([ls[x], ls[y]])
    return a


print(list_sum([10, 20, 30, 40, 50, 60], 50))


def freq_dec(ls):
    dec = {}
    for x in ls:
        dec[x] = dec.get(x, 0) + 1
    return dec


print(freq_dec([1, 1, 1, 2, 3, 3, 4, 4]))
