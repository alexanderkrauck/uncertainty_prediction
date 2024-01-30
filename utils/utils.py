

def flatten_dict(d, parent_key='', sep='/'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

# Example usage
nested_dict = {'a': {'b': {'c': 1, 'd': 2}, 'e': 3}, 'f': 4}
flat_dict = flatten_dict(nested_dict)
print(flat_dict)

def make_lists_strings_in_dict(d):
    for k,v in d.items():
        if isinstance(v, list):
            d[k] = str(v)
    return d