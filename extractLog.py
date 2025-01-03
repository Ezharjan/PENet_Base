import re
import os
import sys

def extract_summary_from_log(log_file_path):
    summaries = []

    with open(log_file_path, 'r') as file:
        lines = file.readlines()

    # Search for "Summary of val round" and capture the next 17 lines
    i = 0
    while i < len(lines):
        if "Summary of val round" in lines[i]:
            summary = ''.join(lines[i:i + 17])
            summaries.append(summary)
            i += 17  # Skip captured lines
        else:
            i += 1

    return summaries

def save_summaries_to_file(summaries, output_file_path):
    with open(output_file_path, 'w') as file:
        for summary in summaries:
            file.write(summary)
            file.write('\n\n')

def main():
    # Check for input arguments
    if len(sys.argv) < 2:
        input_file_path = input("Please enter the input file path: ")
    else:
        input_file_path = sys.argv[1]
    
    # Determine output file path
    if len(sys.argv) < 3:
        output_file_path = os.path.join(os.getcwd(), os.path.basename(input_file_path))
    else:
        output_file_path = sys.argv[2]

    # Extract summaries and save them to the output file
    summaries = extract_summary_from_log(input_file_path)
    save_summaries_to_file(summaries, output_file_path)

    print(f"Summaries have been saved to {output_file_path}")

if __name__ == "__main__":
    main()