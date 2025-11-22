def getTranspose(A):
    rows = len(A)
    cols = len(A[0])

    #initializing values of A transpose and determining no. of rows and cols
    At = [[0] * rows for _ in range(cols)]

    for i in range(rows):
        for j in range(cols):
            At[j][i] = A[i][j]

    return At

# test:
# A = [[1, 1, 1, 1],
#     [2, 2, 2, 2],
#     [3, 3, 3, 3]]

# res = getTranspose(A)
# for row in res:
#     print(*row)

