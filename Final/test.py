
def test(s):
    s[0] = 2


if __name__ == '__main__':
    a = [1]
    test(a)
    print(a) 