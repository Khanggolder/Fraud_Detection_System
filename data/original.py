def count_word_freq(input_string):
    # Chuyen thanh chu thuong va tách tu
    words = input_string.lower().split()
    frequency_dict = {}
    
    for w in words:
        if w in frequency_dict:
            frequency_dict[w] += 1
        else:
            frequency_dict[w] = 1
            
    return frequency_dict

text_data = "apple orange apple banana orange apple"
print(count_word_freq(text_data))