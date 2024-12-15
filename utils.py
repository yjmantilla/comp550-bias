import numpy as np
def get_closest_identity(answer,words,identities):
    """
    Get the score of the answer based on the words and identities
    words: dictionary with words as keys and valences as values
    identities: dictionary with identities as keys and valences as values
    answer: string with the answer
    words_idx: dictionary with words as keys and indexes as values
    identities_idx: dictionary with identities as keys and indexes as values
    valences_idx: dictionary with valences as keys and indexes as values
    """
    answer = answer.lower()
    distances={}
    for w in [x.lower() for x in words]:
        try:
            distances[w] ={}
            for id in [ y.lower() for y in identities]:
                try:
                    distances[w][id] = answer[answer.index(w):].index(id)
                except:
                    distances[w][id] = np.inf # if not found, set to infinity
        except:
            print(f'{w} not found in content {answer}')

    # For each word, get the identity that is closest to it
    closest_identities = {w: min(distances[w], key=distances[w].get) for w in distances.keys()}

    return closest_identities
