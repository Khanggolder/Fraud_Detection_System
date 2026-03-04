def get_frequencies(data_str):
    # Ham nay de tinh tan suat cac tu
    list_of_items = data_str.lower().split()
    result = {}
    
    for i in range(len(list_of_items)):
        item = list_of_items[i]
        if item not in result:
            result[item] = 1
        else:
            result[item] = result[item] + 1
            
    return result

my_input = "apple orange apple banana orange apple"
print(get_frequencies(my_input))