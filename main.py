import timeit


def set_case(sentence):
	return set(sentence) == 26


def iterative_case(sentence):
	appeared_letters = []
	for el in sentence:
		if el not in appeared_letters:
			appeared_letters.append(el)
		else:
			return False
	return True


sentence = 'qwerttyuioplkjhgfdsazxcvbnm'

set_case_time = timeit.timeit(lambda: set_case(sentence), number=10000)
iterative_case_time = timeit.timeit(lambda: iterative_case(sentence), number=10000)

print(f"Time with set: {set_case_time:.6f}")
print(f"Time with iterating: {iterative_case_time:.6f}")
