def count_categories(file_path):
    category_count = {}

    with open(file_path, 'r') as file:
        for line in file:
            # Assuming the category is always the second last word in the line
            category = line.strip().split()[-3]

            if category in category_count:
                category_count[category] += 1
            else:
                category_count[category] = 1

    return category_count

# Replace 'your_file.txt' with the path to your text file
file_path = 'all_valid_anno_info.txt'
counts = count_categories(file_path)
print(counts)