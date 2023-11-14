import re

# test with basic nodes
with open('wavelet_nodes.py', 'r') as file:
    script_content = file.read()

class_pattern = r"class\s+(\w+)\([^)]*\):\s*.*?(?=\s*class\s+(\w+)\([^)]*\)|\Z)"

# all node definitions
node_matches = re.finditer(class_pattern, script_content, re.DOTALL)

# regex for inputs, properties, options, arguments, and outputs
input_pattern = r"self\.add_input\('([^']*)'"
property_pattern = r"self\.add_property\('([^']*)'"
option_pattern = r"self\.add_option\('([^']*)'"
output_pattern = r"self\.add_output\('([^']*)'"


output_filename = 'wavelet_nodes_info.txt'

with open(output_filename, 'w') as output_file:
    for match in node_matches:
        class_name = match.group(1)  # node name
        input_matches = re.findall(input_pattern, match.group(0))  # input name
        property_matches = re.findall(property_pattern, match.group(0))  # property name
        option_matches = re.findall(option_pattern, match.group(0))  # option name
        output_matches = re.findall(output_pattern, match.group(0))

        output_file.write(f"Class: {class_name}\n")
        output_file.write(f"Node(s): nodes here\n")
        output_file.write(f"description:\n")
        output_file.write(f"\tNode description goes here\n")

        output_file.write(f"\ninputs:\n")
        for input_match in input_matches:
            output_file.write(f"\t{input_match} : data_type\n")

        output_file.write(f"\nproperties:\n")
        for property_match in property_matches:
            output_file.write(f"\t{property_match} : property_type : property_description\n")

        output_file.write(f"\noptions:\n")
        for option_match in option_matches:
            output_file.write(f"\t{option_match} : option_type : option_description\n")

        output_file.write(f"\noutput:\n")
        for output_match in output_matches:
            output_file.write(f"\t{output_match} : output_description\n")

        output_file.write("\n" + '-' * 40 + "\n")

        output_file.write("\n")
