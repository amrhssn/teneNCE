import json


def print_dictionary(nested_dict):
    # Convert the nested dictionary to a JSON string with indentation
    pretty_json_str = json.dumps(nested_dict, indent=4)

    # Print the formatted JSON string
    print(pretty_json_str)