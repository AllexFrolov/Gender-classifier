def findMaxSubArray(array: list) -> list:
    ms_start_index = 0
    ms_end_index = 0
    max_sum = array[0]

    start_index = 0
    end_index = 0
    current_sum = array[0]
    for index, val in enumerate(array[1:], 1):
        if current_sum + val > abs(current_sum):
            current_sum += val
            end_index = index
        else:
            if max_sum <= current_sum:
                ms_start_index = start_index
                ms_end_index = end_index
                max_sum = current_sum

            if current_sum <= 0:
                current_sum = val
                start_index = index
                end_index = index
            else:
                current_sum += val

    if max_sum <= current_sum:
        ms_start_index = start_index
        ms_end_index = end_index

    return array[ms_start_index: ms_end_index + 1]


def test(index, array, answer):
    out = findMaxSubArray(array)
    if out == answer:
        print(index, 'pass')
    else:
        print(f'{index}, in: {array}, out: {out}, true: {answer}')


if __name__ == '__main__':
    test(1, [1], [1])
    test(2, [-2, 2], [2])
    test(3, [-5, -1, -2], [-1])
    test(4, [2, 3, 4], [2, 3, 4])
    test(5, [-2, 1, -3, 4, -1, 2, 1, -5, 4], [4, -1, 2, 1])
    test(6, [1, 2, 3, -2, 2, 1], [1, 2, 3, -2, 2, 1])
    test(7, [1, 2, 3, -100, 2, 1, 5], [2, 1, 5])
    test(8, [1, 2, 3, -4, 2, -1, 5], [1, 2, 3, -4, 2, -1, 5])
    test(9, [1, 2, 3, -5, 10, -100, 2, 4, -5, 20], [2, 4, -5, 20])
    test(10, [-2, 1, -3, 4, 1, 2, 1, -5, 4], [4, 1, 2, 1])
    test(11, [2, -1, 3, -3, 4, -4, 5], [2, -1, 3, -3, 4, -4, 5])
    test(12, [2, -1, 3, -3, 4, -4, 5, -100, 1, 2, 3, -5, 10, -100, 2, 4, -5, 20], [2, 4, -5, 20])
    test(13, [-1, -1, -1, -1, -1], [-1])
    test(14, [0], [0])
    test(15, [-1, 0], [0])





