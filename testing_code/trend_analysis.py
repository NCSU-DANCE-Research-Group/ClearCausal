import pymannkendall as mk


def mk_test(data):
    # only use the last 3 values from the data
    data = data[-4: -2]
    result = mk.original_test(data)
    print(f"original data: {data}")
    print(result)


if __name__ == '__main__':
    # simple example test
    # mk_test([1, 2, 3, 4, 5])
    # mk_test([5, 4, 3, 2, 1])
    # mk_test([1, 1, 1, 1, 1])
    # mk_test([1, 2, 1, 2, 1])
    checkout = [38523.87, 50327.65, 51768.71, 46879.83, 35866.2, 59342.86, 17648.5, 90638.17, 51156.21, 41870.8, 38733.1, 0.0, 287040.38]
    mk_test(checkout)
    frontend =  [7, 6, 5, 6, 6, 6, 6, 6, 6, 5, 4, 4, 3]
    mk_test(frontend)
    email = [ 0, 0, 0, 0, 0, 0, 1, 0, -3, 0, 17, 30, 20]
    mk_test(email)